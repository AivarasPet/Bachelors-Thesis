import os
import time
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import json
import tensorflow as tf

work_dir = './cassava-leaf-disease-classification/'
# Load test data
test_df = pd.read_csv('./cassava-leaf-disease-classification/test.csv')

# Image preprocessing
img_width, img_height = 350, 350
batch_size = 32

# Importing the json file with labels
with open(work_dir + 'label_num_to_disease_map.json') as f:
    real_labels = json.load(f)
    real_labels = {int(k):v for k,v in real_labels.items()}
    
# Defining the working dataset
test_df['class_name'] = test_df['label'].map(real_labels)

# Initialize an empty DataFrame to store metrics for all models
model_metrics_df = pd.DataFrame(columns=['Model', 'Parameters', 'Model size (MB)', 'Accuracy', 'Precision', 'Recall', 'F1-score'])


test_datagen = ImageDataGenerator(
    preprocessing_function = tf.keras.applications.efficientnet.preprocess_input,
)

test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_df,
    directory='./cassava-leaf-disease-classification/test_set',
    x_col='image_id',
    y_col='class_name',
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False,
    interpolation='nearest',
)

from tensorflow_addons.losses import SigmoidFocalCrossEntropy


# Test all models in the folder
model_folder = './trained_models'

for root, _, files in os.walk(model_folder):
    for model_file in files:
        if model_file.endswith('.h5'):
            start_time = time.time()

            model_path = os.path.join(root, model_file)
            model = load_model(model_path, custom_objects={'SigmoidFocalCrossEntropy': SigmoidFocalCrossEntropy})

            num_params = model.count_params()
            model_size = os.path.getsize(model_path) / (1024 * 1024)  # in MB

            # Predict and calculate metrics
            y_pred = np.argmax(model.predict(test_generator), axis=1)
            y_true = test_generator.classes
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, average='weighted')
            recall = recall_score(y_true, y_pred, average='weighted')
            f1 = f1_score(y_true, y_pred, average='weighted')
            time_per_image = (time.time() - start_time) / len(y_true)
            total_time = time.time() - start_time

            cm = confusion_matrix(y_true, y_pred)

            # Write results to file
            output_file = f'metrics_{model_file[:-3]}.txt'
            with open(output_file, 'w') as f:
                f.write(f'Model: {model_file}\n')
                f.write(f'Parameters: {num_params}\n')
                f.write(f'Model size: {model_size:.2f} MB\n')
                f.write(f'Time per image: {time_per_image:.5f} s\n')
                f.write(f'Total time: {total_time:.5f} s\n')
                f.write(f'Test dataset size: {len(y_true)} images\n')
                f.write(f'Accuracy: {accuracy:.5f}\n')
                f.write(f'Precision: {precision:.5f}\n')
                f.write(f'Recall: {recall:.5f}\n')
                f.write(f'F1-score: {f1:.5f}\n\n')
                class_accuracy = cm.diagonal() / cm.sum(axis=1)
                # Write accuracy for each class to file
                for i, acc in enumerate(class_accuracy):
                    f.write(f'Accuracy for class {i}: {acc*100:.2f}%\n')

            
            # Add metrics to the DataFrame
            model_metrics_df = model_metrics_df.append({
                'Model': model_file,
                'Parameters': num_params,
                'Model size (MB)': model_size,
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1-score': f1
            }, ignore_index=True)

            # Confusion matrix
            plt.figure(figsize=(10, 10))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=test_generator.class_indices.keys(), yticklabels=test_generator.class_indices.keys())
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title(f'Confusion Matrix for {model_file}')
            plt.savefig(f'confusion_matrix_{model_file[:-3]}.png')
            plt.close()

# Save the DataFrame with metrics to a CSV file
model_metrics_df.to_csv('model_metrics.csv', index=False)

