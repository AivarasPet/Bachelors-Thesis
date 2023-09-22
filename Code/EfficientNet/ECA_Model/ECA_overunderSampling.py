#Built on top of https://www.kaggle.com/code/andreshg/cassava-disease-keras-tf-efficentnet-90/comments

import os
import glob
import random
import shutil
import warnings
import json
import numpy as np
import pandas as pd
from collections import Counter
import keras
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from sklearn.model_selection import train_test_split
from PIL import Image

# Defining the working directories
work_dir = './cassava-leaf-disease-classification/'
train_path = work_dir + 'train_set'
validation_path = work_dir + 'validation_set'
trainCsv = work_dir + 'train.csv'
validationCsv = work_dir + 'validation.csv'
model_name="ECA_overunder_None_1DConv/ECA_overunder_None_1DConv.h5"
final_model_name="ECA_overunder_None_1DConv/final_ECA_overunder_None_1DConv.h5"
metrics_file_path = 'ECA_overunder_None_1DConv/ECA_overunder_None_1DConv_metrics.txt'
EPOCHS = 8
weights_param = None #None 'imagenet'
IMG_SIZE = 350
size = (IMG_SIZE,IMG_SIZE)
n_CLASS = 5
BATCH_SIZE = 15
continue_training_existing_model = False

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
with tf.device('/GPU:0'):
    print('Yes, there is GPU')
    
tf.debugging.set_log_device_placement(True)

# Lets set all random seeds

def seed_everything(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'

seed = 21
seed_everything(seed)
warnings.filterwarnings('ignore')

"""# 1. Data loading ??"""

trainDF = pd.read_csv(work_dir + 'train.csv')
trainDF = trainDF[:100]
validationDF = pd.read_csv(work_dir + 'validation.csv')
validationDF = validationDF[:50]
print(trainDF['label'].value_counts()) # Checking the frequencies of the labels

# Importing the json file with labels
with open(work_dir + 'label_num_to_disease_map.json') as f:
    real_labels = json.load(f)
    real_labels = {int(k):v for k,v in real_labels.items()}
    

real_labels

"""# 2. Data generator ??"""

import pandas as pd
import numpy as np
import os
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split



# Apply undersampling
undersampler = RandomUnderSampler(sampling_strategy='majority')
X_under, y_under = undersampler.fit_resample(trainDF[['image_id']], trainDF['label'])

# Apply oversampling
oversampler = RandomOverSampler(sampling_strategy='not majority')
X_balanced, y_balanced = oversampler.fit_resample(X_under, y_under)

# Combine image_id and labels into a balanced dataframe
balanced_trainDF = pd.DataFrame(np.column_stack((X_balanced, y_balanced)), columns=['image_id', 'label'])

# Defining the working dataset
balanced_trainDF['class_name'] = balanced_trainDF['label'].map(real_labels)
trainDF['class_name'] = trainDF['label'].map(real_labels)
validationDF['class_name'] = validationDF['label'].map(real_labels)

# Group the train_data_balanced DataFrame by image_id and count the occurrences
grouped_data = balanced_trainDF.groupby(['image_id', 'class_name']).size().reset_index(name='count')

# Sort the grouped_data DataFrame by descending count
sorted_grouped_data = grouped_data.sort_values('count', ascending=False)

# Save the sorted_grouped_data DataFrame to a text file
with open('train_data_balanced_sorted.txt', 'w') as f:
    for index, row in sorted_grouped_data.iterrows():
        f.write(f"{row['image_id']} {row['class_name']} {row['count']}\n")


counts = balanced_trainDF['class_name'].value_counts()

# create a new text file and write the counts to it
with open('counts.txt', 'w') as f:
    for index, count in counts.iteritems():
        f.write(f'{index}: {count}\n')


counts2 = trainDF['class_name'].value_counts()

# create a new text file and write the counts to it
with open('countsog.txt', 'w') as f:
    for index, count in counts2.iteritems():
        f.write(f'{index}: {count}\n')


# Define ImageDataGenerator for data augmentation
datagen_train = ImageDataGenerator(
    preprocessing_function = tf.keras.applications.efficientnet.preprocess_input,
    rotation_range = 40,
    width_shift_range = 0.2,
    height_shift_range = 0.2,
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = True,
    vertical_flip = True,
    fill_mode = 'nearest',
)

datagen_val = ImageDataGenerator(
    preprocessing_function = tf.keras.applications.efficientnet.preprocess_input,
)

# Generate new train and validation sets
train_set = datagen_train.flow_from_dataframe(
    dataframe = balanced_trainDF,
    directory=train_path,
    seed=42,
    x_col='image_id',
    y_col='class_name',
    target_size = size,
    class_mode='categorical',
    interpolation='nearest',
    shuffle = True,
    batch_size = BATCH_SIZE,
)

validation_set = datagen_val.flow_from_dataframe(
    dataframe=validationDF,
    directory=validation_path,
    seed=42,
    x_col='image_id',
    y_col='class_name',
    target_size = size,
    class_mode='categorical',
    interpolation='nearest',
    shuffle=True,
    batch_size=BATCH_SIZE,
)

"""# 3. Modeling part ??"""

from keras.models import Sequential
from keras.layers import GlobalAveragePooling2D, Flatten, Dense, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from efficientnet1D import EfficientNetB0
from datetime import datetime
from keras.callbacks import Callback
class MetricsLogger(Callback):
    def __init__(self, filepath):
        super().__init__()
        self.filepath = filepath
        self.metrics = []

    def on_epoch_end(self, epoch, logs=None):
        metrics = [str(logs.get(metric)) for metric in self.model.metrics_names]
        with open(self.filepath, 'a') as f:
            stringas = ""
            for metric in metrics:
                stringas = stringas + metric + ', '  
            stringas = stringas + str(datetime.now()) 
            f.write(stringas + '\n')

def create_model():
    
    model = Sequential()
    # initialize the model with input shape
    model.add(
        EfficientNetB0(
            input_shape = (IMG_SIZE, IMG_SIZE, 3), 
            include_top = False,
            weights=weights_param,
            drop_connect_rate=0.6,
        )
    )
    model.add(GlobalAveragePooling2D())
    model.add(Flatten())
    model.add(Dense(
        256, 
        activation='relu', 
        bias_regularizer=tf.keras.regularizers.L1L2(l1=0.01, l2=0.001)
    ))
    model.add(Dropout(0.5))
    model.add(Dense(n_CLASS, activation = 'softmax'))
    
    return model

leaf_model = create_model()
leaf_model.summary()

STEP_SIZE_TRAIN = train_set.n // train_set.batch_size
STEP_SIZE_TEST = validation_set.n // validation_set.batch_size

"""## Fit the model"""

def model_fit():
    leaf_model = create_model()
    
    # Loss function 
    # https://www.tensorflow.org/api_docs/python/tf/keras/losses/CategoricalCrossentropy
    loss = tf.keras.losses.CategoricalCrossentropy(
        from_logits = False,
        label_smoothing=0.0001,
        name='categorical_crossentropy'
    )
    
    # Compile the model
    leaf_model.compile(
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss = loss, #'categorical_crossentropy'
        metrics = ['categorical_accuracy']
    )
    
    # Stop training when the val_loss has stopped decreasing for 3 epochs.
    # https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/EarlyStopping
    es = EarlyStopping(
        monitor='val_loss', 
        mode='min', 
        patience=3,
        restore_best_weights=True, 
        verbose=1,
    )
    
    # Save the model with the minimum validation loss
    # https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/ModelCheckpoint
    checkpoint_cb = ModelCheckpoint(
        model_name,
        save_best_only=True,
        monitor='val_loss',
        mode='min',
    )
    
    # Reduce learning rate once learning stagnates
    # https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/ReduceLROnPlateau
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=2,
        min_lr=1e-6,
        mode='min',
        verbose=1,
    )

    if continue_training_existing_model:
        leaf_model.load_weights(model_name)
    
    metrics_logger = MetricsLogger(metrics_file_path)

    # Fit the model
    history = leaf_model.fit(
        train_set,
        validation_data=validation_set,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        steps_per_epoch=STEP_SIZE_TRAIN,
        validation_steps=STEP_SIZE_TEST,
        callbacks=[es, checkpoint_cb, reduce_lr, metrics_logger],
    )
    
    # Save the model
    leaf_model.save(final_model_name)  
    
    return history

#sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))

#from tensorflow.compat.v1.keras import backend as K
#K.set_session(sess)

try:
    final_model = keras.models.load_model(final_model_name)
except Exception as e:
    with tf.device('/GPU:0'):
        results = model_fit()
    print('Train Categorical Accuracy: ', max(results.history['categorical_accuracy']))
    print('Test Categorical Accuracy: ', max(results.history['val_categorical_accuracy']))# -*- coding: utf-8 -*-
