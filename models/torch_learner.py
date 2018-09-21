import os

import torch
import torchvision
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from pathlib import Path

from torchsummary import summary
import matplotlib.pyplot as plt

from tqdm import tqdm
import random
from glob import glob
import pandas as pd
import shutil
from collections import Counter

'''
    Utility functions for class
'''

def get_class_from_col(self, col_name):
    dictionary = {
        "NV": "melanocytic_nevi",
        "MEL": "melanoma",
        "AKIEC": "actinic_keratosis",
        "BCC": "BCC",
        "BKL": "benign_keratosis",
        "DF": "dermatofibroma",
        "VASC": "vascular_lesion"
    }
    return dictionary[col_name]


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class Block(nn.Module):

    def __init__():
        super(Block, self).__init()

class Learner(nn.Module):

    def __init__(self):
        super(Learner, self).__init__()

        # if(img_size[0] > 100):
        #     self.conv1 = conv3x3(in_planes, out_planes, stride=1)


        # self.layer1 = nn.Sequential(
        #     nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=2),
        #     nn.BatchNorm2d(16),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=2, stride=2))
        # self.layer2 = nn.Sequential(
        #     nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
        #     nn.BatchNorm2d(32),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=2, stride=2))
        # self.fc = nn.Linear(8*8*32, num_classes)

        # self.num_classes = num_classes
        # self.init_device()

    def create_model(self, num_classes=7, img_size=(224,224), custom_model=None):
        if not custom_model:
            self.img_width, self.img_height = img_size
            
            self.layer1 = nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=2),
                nn.BatchNorm2d(16),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2))
            self.layer2 = nn.Sequential(
                nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2))
            self.fc = nn.Linear(8*8*32, num_classes)

            self.num_classes = num_classes
            self.init_device()

    def prebuilt_model(self, num_classes=7, img_size=(224,224), model=None, freeze_layers=False):
        
        if not model:
            raise ValueError("Model is missing as a function parameter")

        if freeze_layers:
            print("Freezing layers until fully connected layer")
            for param in model.parameters():
                param.requires_grad = False

        self.features = nn.Sequential(*list(model.children())[:-1])
        last_layer = (list(model.children())[-1])

        self.num_classes = num_classes
        self.fc = nn.Linear(in_features=last_layer.in_features, out_features=num_classes, bias=True)

        self.img_width, self.img_height = img_size
        self.init_device()



    def forward(self, x):
        out = self.features(x)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out

    def get_summary(self):
        print(self.img_height, self.img_width)
        summary(self, (3, self.img_height, self.img_width))

    def init_device(self):
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        
        if torch.cuda.is_available():
            print("Using GPU")
            self.cuda()
        else:
            print("Using CPU")

    def check_data_leakage(self):
        trn_images = glob(f"{self.train_dataset.dataset_path}/*/*.jpg")
        test_images = glob(f"{self.test_dataset.dataset_path}/*/*.jpg")
        print(f"Found {len(trn_images)} training images and {len(test_images)} test images")
        trn_images_names = [path.split("/")[-2]+"/"+path.split("/")[-1][:-4] for path in trn_images]
        test_images_names = [path.split("/")[-2]+"/"+path.split("/")[-1][:-4] for path in test_images]

        intersection = list(set(trn_images_names) & set(test_images_names))

        if len(intersection) > 0:
            print(f"There is contamination. The train and test folders contain {len(intersection)} number of the same images")
            self.data_leakage_list = intersection
            print("Run clean_data_leakage(train/test) to remove images")
        else:
            self.data_leakage_list = []
            print("There is no contamination!")

    def clean_data_leakage(self, method=None):
        if hasattr(self, "data_leakage_list") and len(self.data_leakage_list) > 0:
            if method is None:
                raise TypeError("The 'method' parameter has to be passed and set to train/test/random to proceed with cleaning")
            else:
                if method == "train":
                    subdir = self.train_dataset.dataset_path
                    print("clean train folder")
                elif method == "test":
                    subdir = self.test_dataset.dataset_path
                    print("clean test folder")
                elif method == "random":
                    print("clean randomly from both folders")

                for img in self.data_leakage_list:
                    os.remove(f"{subdir}/{img}.jpg")
                    print(f"Removed image: {subdir}/{img}.jpg")
        else:
            raise TypeError("There is no data in list self_leakage_list and it is undefined. Run test_data_leakage first and then try again.")

    

    def copy_from_folder(self, img_folder, csv_path=None, train=True):
        subdir = "train" if train else "test"

        if csv_path:
            images = glob(f"{img_folder}{subdir}/*.jpg")

            print(f"Found {len(images)} training images to copy")
            ground_t = pd.read_csv(csv_path)

            cols = ground_t.columns.tolist()

            no_of_imgs = 0
            _dict = {}
            for i, row in tqdm(enumerate(ground_t.iterrows())):
                if i >= 1:
                    val = [col for col in row[1]]
                    idx = val.index(1.0)
                    class_type = get_class_from_col(cols[idx])
                    img_name = f"{val[0]}.jpg"

                    if os.path.exists(f"{self.test_path}/{class_type}/{img_name}"):
                        print("Not copying, image exists already in test folder")
                    else:
                        shutil.copy(f"{img_folder}{subdir}/{img_name}", f"{self.trn_path}/{class_type}/{img_name}")
            print("Done copying images!")
        else:
            raise ValueError(
                "Use a CSV File - folder structure not currently implemented")
            # TODO: If not CSV - use the classes in the train folder, copy over - the folder structure in the
            # copied folder must be the same..

    def load_datasets(self, train_dataset, test_dataset, batch_size=64):
        
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.classes = train_dataset.dataset.classes
        self.batch_size = batch_size
        self.no_of_train_images = len(train_dataset.index)
        self.no_of_test_images = len(test_dataset.index)

        if train_dataset.subset and test_dataset.subset:
            test_subSampler = SubsetRandomSampler(test_dataset.index)
            train_subSampler = SubsetRandomSampler(train_dataset.index)

            self.test_loader = torch.utils.data.DataLoader(dataset=self.test_dataset, sampler=test_subSampler, batch_size=batch_size, num_workers=4)
            self.train_loader = torch.utils.data.DataLoader(dataset=self.train_dataset, sampler=train_subSampler, batch_size=batch_size, num_workers=4)
        elif train_dataset.subset and not test_dataset.subset:
            print("Train dataset subset is true and not test dataset subset")
            train_subSampler = SubsetRandomSampler(train_dataset.index)
           
            self.train_loader = torch.utils.data.DataLoader(dataset=self.train_dataset, sampler=train_subSampler, batch_size=batch_size, num_workers=4)
            self.test_loader = torch.utils.data.DataLoader(dataset=self.test_dataset, batch_size=batch_size, num_workers=4)

        elif not train_dataset and test_dataset.subset:
            print("Test dataset subset is true and not train dataset subset")
            test_subSampler = SubsetRandomSampler(test_dataset.index)
            self.test_loader = torch.utils.data.DataLoader(dataset=self.test_dataset, sampler=test_subSampler, batch_size=batch_size, num_workers=4)
            self.train_loader = torch.utils.data.DataLoader(dataset=self.train_dataset, batch_size=batch_size, num_workers=4)
            
        else:
            # load the whole dataset
            self.train_loader = torch.utils.data.DataLoader(dataset=self.train_dataset, batch_size=batch_size, shuffle=True)
            self.test_loader = torch.utils.data.DataLoader(dataset=self.test_dataset, batch_size=batch_size, shuffle=False)

        print(f"Done loading Image data. Got {len(train_dataset)} training images and {len(test_dataset)} test images")
        if train_dataset.subset:
            print(f"Training dataset subset is set to {train_dataset.subset} and {round(train_dataset.subset_percentage*len(train_dataset))} no. of total images were loaded.")
            print(f"Testing dataset subset is set to {test_dataset.subset} and {round(test_dataset.subset_percentage*len(test_dataset))} no. of total images were loaded.")
      
    def move_train_to_test(self, percentage, class_type, all_classes=None):
        if all_classes:
            print(f"move {percentage} of data from train to test for all classes in train folder")
            print("Need to implement that function")
        else:
            print(f"{self.trn_path}/{class_type}/*.jpg")
            paths = glob(f"{self.trn_path}/{class_type}/*.jpg")
            no_imgs_to_move = round(percentage*len(paths))
            print(f"found {len(paths)} photos, moving {percentage*100}% of them({no_imgs_to_move} images)")
            random.shuffle(paths)
            imgs_to_mov = paths[:no_imgs_to_move]
            for image_path in imgs_to_mov:
                img_name = image_path.split("/")[-1]
                shutil.move(image_path, f"{self.test_path}/{class_type}/{img_name}")
        print("Done moving images")

    def show_images(self, figsize=(15,15), dataset='train'):
        fig, axes = plt.subplots(3, 3, figsize=figsize)

        print(f"Showing images from {dataset} dataset. To change, set the 'dataset' parameter")
        
        data_iter = iter(self.train_loader) if dataset == "train" else iter(self.test_loader)
        
        _images, labels, index, path = data_iter.next()
        images = _images.numpy().transpose(0, 2, 3, 1)

        for i, ax in enumerate(axes.flat):
            # plot img
            ax.imshow(images[i, :, :, :], aspect='auto')
            # show true & predicted classes
            true_label = labels[i]
            # if cls_pred is None:
            xlabel = "{}, ({}), filename: {}".format(self.train_dataset.dataset.classes[true_label], true_label, path[i].split("/")[-1])
            # else:
            #     cls_pred_name = labels[cls_pred[i]]
            #     xlabel = "True: {0}\nPred: {1}".format(
            #         cls_true_name, cls_pred_name
            #     )
            ax.set_xlabel(xlabel)
            ax.set_xticks([])
            ax.set_yticks([])

        plt.show()

    def train(self, epochs=5, lr=0.001, show_loss_every_step=False):

        if self.num_classes > 2:
            # Loss and optimizer
            self.criterion = "Cross Entropy Loss"
            criterion = nn.CrossEntropyLoss()
        else:
            self.criterion = "Binary Cross Entropy Loss"

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        self.lr = lr
        self.epochs = epochs

        if self.epochs:
            print("-"*25)
            print(f"Training with parameters: ")
            print(f"Learning_rate: {self.lr}")
            print(f"batch_size: {self.batch_size}")
            print(f"No of training images: {self.no_of_train_images}")
            print(f"Number of epochs: {self.epochs}")
            print(f"No of output classes: {self.num_classes}")
            print(f"Criterion: {self.criterion}")
            print("-"*25)

        loss_array = []

        # Train the model
        total_step = len(self.train_loader)
        lowest_loss = 100
        # each epoch
        for epoch in range(epochs):
            # each batch
            cum_loss = 0
            for i, (images, labels, path, index) in enumerate(self.train_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)

                # Forward pass
                outputs = self(images)
                loss = criterion(outputs, labels)

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if loss.item() < lowest_loss:
                    lowest_loss = loss.item()

                cum_loss += loss.item()
                loss_array.append(loss.item())

            if show_loss_every_step:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch + 1, epochs, i+1, total_step, (cum_loss/total_step)))

                if (i) % (self.no_of_train_images // self.batch_size) == 0:
                    print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, epochs, i+1, total_step, (cum_loss/total_step)))
                # if (i) % (1) == 0:
        if lowest_loss != 100:
            print(f"Done training - Lowest loss value: {lowest_loss}")
            print("Loss array:")
            print(loss_array)

    def test(self):
        # Test the model
        print(f"Testing on {self.no_of_test_images} testing images")
        self.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)

        with torch.no_grad():
            correct = 0
            total = 0
            for i, (images, labels, path, index) in tqdm(enumerate(self.test_loader)):
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            print('Test Accuracy of the model on the {} test images: {} %'.format(self.no_of_test_images, round(100 * correct / total, 2)))
