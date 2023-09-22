import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from torchvision import transforms
from PIL import Image
import json

from efficient_vit import EfficientViT

model_path = './Effnet_Transformer/models/checkpoint_effnet_transformer.pth'


def check_correct(preds, labels):
    preds = preds.cpu()
    labels = labels.cpu()
    y_pred_softmax = torch.softmax(preds, dim=1)
    predicted_classes = torch.argmax(y_pred_softmax, dim=1)

    # Convert the tensor to a NumPy array
    predicted_classes_array = predicted_classes.numpy()

    correct = 0
    positive_class = 0
    negative_class = 0
    for i in range(len(labels)):
        pred = int(predicted_classes_array[i])
        if labels[i] == pred:
            correct += 1
        if pred == 1:
            positive_class += 1
        else:
            negative_class += 1
    return correct, positive_class, negative_class


class CustomDataset(Dataset):
    def __init__(self, dataframe, directory, transform=None, target_size=(224, 224)):
        self.dataframe = dataframe
        self.directory = directory
        # self.preprocess_function = preprocess_function
        self.transform = transform
        self.target_size = target_size
        self.class_names = sorted(list(dataframe['class_name'].unique()))
        self.class_name_to_label = {class_name: i for i, class_name in enumerate(self.class_names)}

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = self.dataframe.iloc[idx]['image_id']
        img_path = f"{self.directory}/{img_name}"
        img = Image.open(img_path).convert('RGB').resize(self.target_size)
        # img = self.preprocess_function(img)
        if self.transform:
            img = self.transform(img)

        class_name = self.dataframe.iloc[idx]['class_name']
        label = self.class_name_to_label[class_name]
        label = torch.tensor(label, dtype=torch.long)

        return img, label



import pandas as pd
import os 
import yaml 

def main():
    # Defining the working directories
    work_dir = './cassava-leaf-disease-classification/'
    test_path = work_dir+'test_set'
    BATCH_SIZE = 5
    MODELS_PATH = "py_models"
    validationDF = pd.read_csv(work_dir + 'test.csv')

    # Importing the json file with labels
    with open(work_dir + 'label_num_to_disease_map.json') as f:
        real_labels = json.load(f)
        real_labels = {int(k):v for k,v in real_labels.items()}

    # Defining the working dataset
    validationDF['class_name'] = validationDF['label'].map(real_labels)

    IMG_SIZE = 224
    size = (IMG_SIZE,IMG_SIZE)
    # Create the custom Dataset and DataLoader
    test_transforms = transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor()
        ])

    test_dataset = CustomDataset(validationDF, test_path, transform=test_transforms, target_size=size)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Get list of model files
    model_files = os.listdir(MODELS_PATH)

    # Create an empty DataFrame to store the results
    results_df = pd.DataFrame(columns=['Model', 'Accuracy'])

    for model_file in model_files:
        if model_file.endswith('.pth'):
            # Load the trained model
            model_path = os.path.join(MODELS_PATH, model_file)
            channels = 1280

            model_file_without_extension = model_file.rstrip(".pth")
            config_path = os.path.join(MODELS_PATH, model_file_without_extension, 'architecture.yaml')

            with open(config_path, 'r') as ymlfile:
                config = yaml.safe_load(ymlfile)

            model_state_dict = torch.load(model_path, map_location=device)

            model = EfficientViT(config=config, channels=channels)
            model.load_state_dict(model_state_dict)
            model.to(device)

            val_correct = 0
            val_counter = 0
            for index, (val_images, val_labels) in enumerate(test_loader):
                val_images, val_labels = val_images.to(device), val_labels.to(device)
                print(val_counter)
                val_pred = model(val_images)
                corrects, positive_class, negative_class = check_correct(val_pred, val_labels)
                val_correct += corrects
                val_counter += 1

            val_accuracy = val_correct/(val_counter*BATCH_SIZE)
            print(f'Test Accuracy of the model {model_file}: {val_accuracy}%')
            
            # Append the results to the DataFrame
            results_df = results_df.append({'Model': model_file, 'Accuracy': val_accuracy}, ignore_index=True)

    # Save the DataFrame to a csv file
    results_df.to_csv('model_accuracies.csv', index=False)

if __name__ == '__main__':
    main()
