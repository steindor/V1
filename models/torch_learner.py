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


class Learner(nn.Module):

    def __init__(self, num_classes=7):
        super(Learner, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(56*56*256, num_classes)

        self.num_classes = num_classes
        self.init_device()

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out

    def get_summary(self):
        summary(self, (3, 224, 224))

    def init_device(self):
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu')
        # self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        print(f"Using {str(self.device)}")

    def check_data_leakage(self):
        trn_path = Path(self.trn_path)
        classes = [str(x).split("/")[-1]
                   for x in trn_path.iterdir() if x.is_dir()]

        trn_images = glob(f"{self.trn_path}/*/*.jpg")
        test_images = glob(f"{self.test_path}/*/*.jpg")
        print(f"Found {len(trn_images)} training images and {len(test_images)} test images")
        trn_images_names = [path.split(
            "/")[-2]+"/"+path.split("/")[-1][:-4] for path in trn_images]
        test_images_names = [path.split(
            "/")[-2]+"/"+path.split("/")[-1][:-4] for path in test_images]

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
                    subdir = self.trn_path
                    print("clean train folder")
                elif method == "test":
                    subdir = self.test_path
                    print("clean test folder")
                elif method == "random":
                    print("clean randomly from both folders")

                for img in self.data_leakage_list:
                    os.remove(f"{subdir}/{img}.jpg")
                    print(f"Removed image: {subdir}/{img}.jpg")
        else:
            raise TypeError("There is no data in list self_leakage_list and it is undefined. Run test_data_leakage first and then try again.")

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
                    class_type = self.get_class_from_col(cols[idx])
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

    def display_train_images(self):
        fig, axes = plt.subplots(3, 3)

        data_iter = iter(self.train_loader)
        _images, labels, path, index = data_iter.next()
        images = _images.numpy().transpose(0, 2, 3, 1)

        for i, ax in enumerate(axes.flat):
            # plot img
            ax.imshow(images[i, :, :, :], interpolation='spline16')

            # show true & predicted classes
            true_label = labels[i]
            # if cls_pred is None:
            xlabel = "{}".format(true_label)
            # else:
            #     cls_pred_name = labels[cls_pred[i]]
            #     xlabel = "True: {0}\nPred: {1}".format(
            #         cls_true_name, cls_pred_name
            #     )
            ax.set_xlabel(xlabel)
            ax.set_xticks([])
            ax.set_yticks([])

        plt.show()

    def train(self, epochs=5, lr=0.001):

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

        # Train the model
        total_step = len(self.train_loader)
        lowest_loss = 100
        # each epoch
        for epoch in range(epochs):
            # each batch
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

                # if (i) % (self.no_of_train_images // self.batch_size) == 0:
                # if (i) % (1) == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(
                    epoch+1, epochs, i+1, total_step, loss.item()))
        print(f"Done training - Lowest loss value: {lowest_loss}")

    def test(self):
        # Test the model
        print(f"Testing on {self.no_of_test_images} testing images")
        self.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)

        print(f"Length of test loader: {len(self.test_loader)}")

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

            print('Test Accuracy of the model on the {} test images: {} %'.format(
                len(self.test_loader), 100 * correct / total))
