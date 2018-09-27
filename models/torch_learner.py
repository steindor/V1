import os

import torch
import torchvision
import torch.nn as nn
import numpy as np
# import torchvision.transforms as transforms
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

from IPython.core.debugger import set_trace

'''
    Utility functions
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

def get_model(model_name, pretrained):

    model_dict = {
        "resnet18": torchvision.models.resnet18(pretrained=pretrained),
        "resnet34": torchvision.models.resnet34(pretrained=pretrained),
        "resnet50": torchvision.models.resnet50(pretrained=pretrained),
        "resnet101": torchvision.models.resnet101(pretrained=pretrained),
        "densenet121": torchvision.models.densenet121(pretrained=pretrained),
        "densenet161": torchvision.models.densenet161(pretrained=pretrained),
        "densenet169": torchvision.models.densenet169(pretrained=pretrained),
        "densenet201": torchvision.models.densenet201(pretrained=pretrained),
        "squeezenet1_0": torchvision.models.squeezenet1_0(pretrained=pretrained),
        "inception_v3": torchvision.models.inception_v3(pretrained=pretrained)
    }
    return model_dict["{}".format(model_name)]

class FeatureExtractor(nn.Module):
    
    def __init__(self, model_name, pretrained=False, freeze=True):
        super(FeatureExtractor, self).__init__()

        self.model_name = model_name

        if "dense" in model_name.lower():
            model = get_model(model_name, pretrained)
            self.features = nn.Sequential(*list(model.features.children()))
        elif "resnet" in model_name.lower():
            model = get_model(model_name, pretrained)
            self.features = nn.Sequential(*list(model.children())[:-1])
        elif "inception" in model_name.lower():
            model = get_model(model_name, pretrained)
            model.aux_logits = False
            del  model._modules['fc']
            model.fc = nn.Linear(2048, 512)
            self.features = model

        # bottleneck_tensor_size = 2048 f inception

        if not model_name:
            raise ValueError("Model name is missing, please pass in model_name parameter")

        # freeze the layers since this is a feature extractor
        for param in model.parameters():
            param.requires_grad = False

    def __sizeof__(self):
        super(FeatureExtractor, self).__sizeof__()

    def forward(self, x):
        x = self.features(x)
        return x

class ModelEnsemble(nn.Module):
    def __init__(self, n_classes):
        super(ModelEnsemble, self).__init__()
        # resnet34
        self.fc1 = nn.Linear(512, 512)
        # densenet 121
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, n_classes)

    def forward(self, inp1, inp2, inp3):
        out1 = self.fc1(inp1)
        out2 = self.fc2(inp2)
        out = out1 + out2
        out = self.fc3(out)

class Learner(nn.Module):
    def __init__(self):
        super(Learner, self).__init__()

    def prebuilt_model(self, num_classes=7, img_size=(224,224), model_name=None, pretrained=False, freeze_all=False):
        
        if not model_name:
            raise ValueError("Model_name is missing as a function parameter")

        self.num_classes = num_classes
        self.model_name = model_name
        self.img_width, self.img_height = img_size
        self.init_device()

        model = get_model(model_name, pretrained=pretrained)

        if freeze_all:
            print("Freezing all layers until classifier")
            for param in model.parameters():
                param.requires_grad = False
        
        self.replace_top_layer(model)

    def forward(self, x):
        out = self.features(x)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out

    def replace_top_layer(self, model):
        if "resnet" in self.model_name:
            self.features = nn.Sequential(*list(model.children())[:-1])
            last_layer = (list(model.children())[-1])
            self.fc = nn.Linear(in_features=last_layer.in_features, out_features=self.num_classes, bias=True)
        elif "inception" in self.model_name:
            self.aux_logits = False
            # Inception bottleneck_features - in 2048
            self._modules['fc'] = nn.Linear(in_features=2048, out_features=self.num_classes, bias=True)
            
    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze():
        for param in self.parameters():
            param.requires_grad = True

    def get_summary(self):
        summary(self, (3, self.img_height, self.img_width))

    def show_training_graph(self):
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(20, 5), squeeze=False)
        ax[0, 0].plot(self.test_loss, label="Test loss")
        ax[0, 0].plot(self.train_loss, label="Training loss")
        ax[0, 0].legend()
        ax[0, 0].set_title("Training / Test loss")
        ax[0, 1].plot(self.test_acc, label="Test accuracy")
        ax[0, 1].plot(self.train_acc, label="Training accuracy")
        ax[0, 1].legend()
        print("Still missing Train accuracy!")

    def init_device(self):      
        if torch.cuda.is_available():
            print("Using GPU")
            self.device = 'cuda:0'
            self.pin_memory = True
            self.cuda()
        else:
            self.device = 'cpu'
            self.pin_memory = False
            print("Using CPU")

    def check_data_leakage(self):

        if not hasattr(self, 'train_dataset'):
            raise ValueError("There are no datasets loaded in the Learner")

        trn_images = glob(f"{self.train_dataset.dataset.root}/*/*.jpg")
        test_images = glob(f"{self.test_dataset.dataset.root}/*/*.jpg")
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

            self.test_loader = torch.utils.data.DataLoader(dataset=self.test_dataset, sampler=test_subSampler, batch_size=batch_size, num_workers=4, pin_memory=self.pin_memory)
            self.train_loader = torch.utils.data.DataLoader(dataset=self.train_dataset, sampler=train_subSampler, batch_size=batch_size, num_workers=4, pin_memory=self.pin_memory)
        elif train_dataset.subset and not test_dataset.subset:
            print("Train dataset subset is true and not test dataset subset")
            train_subSampler = SubsetRandomSampler(train_dataset.index)
           
            self.train_loader = torch.utils.data.DataLoader(dataset=self.train_dataset, sampler=train_subSampler, batch_size=batch_size, num_workers=4, pin_memory=self.pin_memory)
            self.test_loader = torch.utils.data.DataLoader(dataset=self.test_dataset, batch_size=batch_size, num_workers=4, pin_memory=self.pin_memory)

        elif not train_dataset and test_dataset.subset:
            print("Test dataset subset is true and not train dataset subset")
            test_subSampler = SubsetRandomSampler(test_dataset.index)
            self.test_loader = torch.utils.data.DataLoader(dataset=self.test_dataset, sampler=test_subSampler, batch_size=batch_size, num_workers=4, pin_memory=self.pin_memory)
            self.train_loader = torch.utils.data.DataLoader(dataset=self.train_dataset, batch_size=batch_size, num_workers=4, pin_memory=self.pin_memory)
            
        else:
            # load the whole dataset
            self.train_loader = torch.utils.data.DataLoader(dataset=self.train_dataset, batch_size=batch_size, shuffle=True, pin_memory=self.pin_memory)
            self.test_loader = torch.utils.data.DataLoader(dataset=self.test_dataset, batch_size=batch_size, shuffle=False, pin_memory=self.pin_memory)

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

    def fit(self, epochs=5, lr=0.001, show_loss_every_step=False):

        val_losses = []

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
            print(f"No of test images: {self.no_of_test_images}")
            print(f"Number of epochs: {self.epochs}")
            print(f"No of output classes: {self.num_classes}")
            print(f"Criterion: {self.criterion}")
            print("-"*25)

        loss_array = []

        # Train the model
        total_step = len(self.train_loader)
        lowest_loss = 100
        test_acc = []
        test_loss = []
        train_acc = []
        train_loss = []
        # each epoch

        for epoch in tqdm(range(epochs)):
            self.train()
            cum_loss_train = 0
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

                cum_loss_train += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                loss_array.append(loss.item())
                train_acc.append(100 * correct / total)

                if show_loss_every_step:
                    print('Epoch [{}/{}], Training loss: {:.4f}'.format(epoch + 1, epochs, (cum_loss_train/total_step)))



            self.eval()
            with torch.no_grad():

                total = 0
                correct = 0
                wrong = []

                cum_loss_test = 0

                for i, (images, labels, path, index) in enumerate(self.test_loader):
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                    
                    outputs = self(images)
                    loss = criterion(outputs, labels)

                    cum_loss_test += loss.item()

                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    wrong.append(predicted != labels)

            train_loss.append(cum_loss_train / len(self.train_loader))
            test_loss.append(cum_loss_test / len(self.test_loader))
            test_acc.append(100 * correct / total)

            print('Epoch [{}/{}], Training loss: {:.4f} - Training accuracy: {:.2f}% Test Loss: {:.2f} - Test accuracy: {:.2f}%'.format(epoch+1, epochs, (cum_loss_train/total_step), (100*correct/total), cum_loss_test / len(self.test_loader), 100 * correct / total))

        if not hasattr(self, "test_loss"):
            self.test_loss = [] 
            self.train_loss = []
            self.train_acc = []
            self.test_acc = []

        self.test_loss += test_loss
        self.train_loss += train_loss
        self.test_acc += test_acc
        self.train_acc = train_acc

        self.show_training_graph()


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
