import os
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
import json
import collections
from torch.optim import lr_scheduler
import math
import argparse

# Defining the working directories
continueTraining = False
work_dir = './cassava-leaf-disease-classification/'
train_path = work_dir + 'train_set'
validation_path = work_dir + 'validation_set'
trainCsv = work_dir + 'train.csv'
metrics_file_path = './metrics.txt'
validationCsv = work_dir + 'validation.csv'

MODELS_PATH = "models"
checkPointPath = 'checkpoint_effnet_transformer.pth'

trainDF = pd.read_csv(work_dir + 'train.csv')
#trainDF = trainDF[:300]
validationDF = pd.read_csv(work_dir + 'validation.csv')
#validationDF = validationDF[:100]
print(trainDF['label'].value_counts()) # Checking the frequencies of the labels

# Importing the json file with labels
with open(work_dir + 'label_num_to_disease_map.json') as f:
    real_labels = json.load(f)
    real_labels = {int(k):v for k,v in real_labels.items()}
    
# Defining the working dataset
trainDF['class_name'] = trainDF['label'].map(real_labels)
validationDF['class_name'] = validationDF['label'].map(real_labels)

IMG_SIZE = 224
size = (IMG_SIZE,IMG_SIZE)
BATCH_SIZE = 15

import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
from progress.bar import ChargingBar

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

from datetime import datetime
def log_metrics(metrics, logs=None):
        with open(metrics_file_path, 'a') as f:
            stringas = ""
            for metric in metrics:
                stringas = stringas + metric + ', '  
            stringas = stringas + str(datetime.now()) 
            f.write(stringas + '\n')  

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', default=300, type=int,
                        help='Number of training epochs.')
    parser.add_argument('--workers', default=10, type=int,
                        help='Number of data loader workers.')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='Path to latest checkpoint (default: none).')
    parser.add_argument('--dataset', type=str, default='All', 
                        help="Which dataset to use (Deepfakes|Face2Face|FaceShifter|FaceSwap|NeuralTextures|All)")
    parser.add_argument('--max_videos', type=int, default=-1, 
                        help="Maximum number of videos to use for training (default: all).")
    parser.add_argument('--config', type=str, 
                        help="Which configuration to use. See into 'config' folder.")
    parser.add_argument('--efficient_net', type=int, default=0, 
                        help="Which EfficientNet version to use (0 or 7, default: 0)")
    parser.add_argument('--patience', type=int, default=5, 
                        help="How many epochs wait before stopping for validation loss not improving.")
    opt = parser.parse_args()

    import yaml
    config_path = os.path.join(os.path.dirname(__file__), 'configs', 'architecture.yaml')

    with open(config_path, 'r') as ymlfile:
        config = yaml.safe_load(ymlfile)

    # Define the transforms
    train_transforms = transforms.Compose([
        transforms.Resize(size),
        transforms.RandomRotation(40),
        transforms.RandomResizedCrop(size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor()
    ])

    val_transforms = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor()
    ])

    # Create the datasets
    train_dataset = CustomDataset(trainDF, train_path, transform=train_transforms, target_size=size)
    val_dataset = CustomDataset(validationDF, validation_path, transform=val_transforms, target_size=size)

    # Create the data loaders
    train_loader = DataLoader(train_dataset, batch_size=config['training']['bs'], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=config['training']['bs'], shuffle=True, num_workers=4)

    from efficient_vit import EfficientViT
    channels = 1280

    model = EfficientViT(config=config, channels=channels)
    if continueTraining:
        model_state_dict = torch.load( os.path.join(MODELS_PATH,  checkPointPath))
        model.load_state_dict(model_state_dict)

    model.train()

    optimizer = torch.optim.SGD(model.parameters(), lr=config['training']['lr'], weight_decay=config['training']['weight-decay'])
    scheduler = lr_scheduler.StepLR(optimizer, step_size=config['training']['step-size'], gamma=config['training']['gamma'])
    starting_epoch = 0

    # train_counters = collections.Counter(image[1] for image in train_dataset)

    # class_weights = train_counters[0] / train_counters[1]
    import torch.nn.functional as F

    loss_function =  torch.nn.CrossEntropyLoss()#torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor())
    # Set the number of training epochs
    num_epochs = 8
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_samples = len(train_dataset)
    # train_dataset = shuffle_dataset(train_dataset)
    validation_samples = len(val_dataset)
    # validation_dataset = shuffle_dataset(validation_dataset)
    
    if torch.cuda.is_available():
        model = model.cuda()
    counter = 0
    not_improved_loss = 0
    previous_loss = math.inf
    for t in range(starting_epoch, opt.num_epochs + 1):
        if not_improved_loss == opt.patience:
            break
        counter = 0

        total_loss = 0
        total_val_loss = 0
        
        bar = ChargingBar('EPOCH #' + str(t), max=(len(train_loader)*config['training']['bs'])+len(val_loader))
        train_correct = 0
        positive = 0
        negative = 0
        for index, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            #images = np.transpose(images, (0, 3, 1, 2))
            # labels = labels.unsqueeze(1)
            # labels = labels.squeeze()
            if torch.cuda.is_available():
                images = images.cuda()
            y_pred = model(images)
            # y_pred = y_pred.cpu()
            loss = loss_function(y_pred, labels)
        
            corrects, positive_class, negative_class = check_correct(y_pred, labels)  
            train_correct += corrects
            positive += positive_class
            negative += negative_class
            optimizer.zero_grad()
            
            loss.backward()
            
            optimizer.step()
            counter += 1
            total_loss += round(loss.item(), 2)
            
            if index%10 == 0: # Intermediate metrics print
                print("\nLoss: ", total_loss/counter, "Accuracy: ",train_correct/(counter*config['training']['bs']) ,"Train 0s: ", negative, "Train 1s:", positive)


            for i in range(config['training']['bs']):
                bar.next()

        val_correct = 0
        val_positive = 0
        val_negative = 0
        val_counter = 0
        train_correct /= train_samples
        total_loss /= counter
        for index, (val_images, val_labels) in enumerate(val_loader):
            val_images, val_labels = val_images.to(device), val_labels.to(device)
#            val_images = np.transpose(val_images, (0, 3, 1, 2))
            if torch.cuda.is_available():
                val_images = val_images.cuda()
            #val_labels = val_labels.unsqueeze(1)
            val_pred = model(val_images)
            val_loss = loss_function(val_pred, val_labels)
            total_val_loss += round(val_loss.item(), 2)
            corrects, positive_class, negative_class = check_correct(val_pred, val_labels)
            print('corrects: ')
            print(corrects)
            val_correct += corrects
            val_positive += positive_class
            val_counter += 1
            val_negative += negative_class
            bar.next()
            
        scheduler.step()
        bar.finish()
            

        val_accuracy = val_correct/(val_counter*config['training']['bs'])
        total_val_loss /= val_counter
        val_correct /= validation_samples
        if previous_loss <= total_val_loss:
            print("Validation loss did not improved")
            not_improved_loss += 1
        else:
            not_improved_loss = 0
        
        previous_loss = total_val_loss
        
        metrics = [
            str(total_loss), # loss padalintas
            str(val_accuracy) # accuracy
        ]
        log_metrics(metrics)

        #print("#" + str(t) + "/" + str(opt.num_epochs) + " loss:" +
        #    str(total_loss) + " accuracy:" + str(train_correct) +" val_loss:" + str(total_val_loss) + " val_accuracy:" + str(val_accuracy) + " val_0s:" + str(val_negative) + "/" + str(np.count_nonzero(val_labels == 0)) + " val_1s:" + str(val_positive) + "/" + str(np.count_nonzero(val_labels == 1)))



        if not os.path.exists(MODELS_PATH):
            os.makedirs(MODELS_PATH)
        torch.save(model.state_dict(), os.path.join(MODELS_PATH,  checkPointPath))
    torch.save(model.state_dict(), os.path.join(MODELS_PATH,  "efficientnetB"+str(opt.efficient_net)+"_file.pth"))
        

if __name__ == '__main__':
    main()