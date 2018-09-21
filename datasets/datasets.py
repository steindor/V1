from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from glob import glob
from tqdm import tqdm
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import random
import torch

class ImageFolderWithPaths(datasets.ImageFolder):
    """
    Custom dataset class that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """
    # override the __getitem__ method. this is the method dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path

class Derma(Dataset):
    '''
        The primary dataset
        Args:
            batch_size (int): batch size
            dataset_path (string): Path to the dataset
            img_path (string): path to the folder where images are
            img_size (tuple): the image size - format (32,32) which is also default
            test (boolean): set to true if dataset is test set - hard code paths? Implements different transformations
            augment(boolean): adds augmentation to training set
            subset(boolean): set to true if subset of the dataset is preferred
            subset_percentage(float): how large percentage of the dataset gets chosen(0.0 - 1.0)
            normal_distribution(boolean): set to true if preferred to train on a dataset which represents real life distribution as a physician would encounter cases
            distribution_dict(dictionary): Not implemented yet - possible to set the classes to different values which sum up to 1 to get the selected distribution in that way
            transform: pytorch transforms for transforms and tensor conversion
    '''

    def __init__(self, batch_size, dataset_path, img_size=(32, 32), shuffle=False, test=False, train_test_split=0.2, augment=False, subset=False, subset_percentage=None, normal_distribution=True, distribution_dict=None):
        self.batch_size = batch_size
        self.dataset_path = dataset_path
        self.train_test_split = train_test_split
        self.img_size = img_size
        self.test = test
        self.shuffle = shuffle
        self.augment = augment
        self.subset = subset
        self.subset_percentage = subset_percentage
        
        if subset_percentage and not subset:
            raise ValueError("Subset is set to false but the percentage is set!")
        elif subset and not subset_percentage:
            raise ValueError("Subset is set to true but the percentage is not set. Choose a value for subset_percentage!")
        if normal_distribution and not distribution_dict:
            print("Getting data with normal distribution - Cases spread as a physician would encounter them")
        elif not normal_distribution and distribution_dict:
            print("Getting data like defined in distribution dict")
            distribution_dict = {
                "melanocytic_nevi": 0.5,
                "melanoma": 0.05
            }
        else:
            print("Getting all data")


        self.dataset = ImageFolderWithPaths(self.dataset_path, transform=self.transform()) 
        if subset:
            self.index = self.get_subsample_indices(shuffle=shuffle, percentage=subset_percentage)
        
        print(f"found {len(self)} images - Subset is set to {self.subset}, creating dataset with {round(self.subset_percentage*len(self))} images")
       

    def __getitem__(self, index):
        data, labels, path = self.dataset[index]
        return data, labels, index, path

    def __len__(self):
        return len(self.dataset)

    def transform(self):
        
        # TODO: Calculate std and mean of dataset to use on the fly
        # https://discuss.pytorch.org/t/normalization-in-the-mnist-example/457/12
        # TODO: Write a function that shows training images with all transformations
        
        normalize = transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5],
        )

        test_transform = transforms.Compose([
            transforms.Resize(size=self.img_size),
            transforms.ToTensor(),
            normalize
        ])

        if self.augment:
            train_transform = transforms.Compose([
                transforms.Resize(size=self.img_size),
                # transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        else:
            train_transform = transforms.Compose([
                transforms.Resize(size=self.img_size),
                transforms.ToTensor(),
                normalize,
            ])
        
        if self.test:
            return test_transform
        else:
            return train_transform

    def get_subsample_indices(self, percentage, shuffle=None):
        self.subsample = True
        self.no_of_test_images = int(np.floor(percentage * len(self)))

        # finna indexes af ollum clossum
        # na i random subsample med percentage * fjolda myndum i hverjum classa fyrir sig?

        r = list(range(len(self) - 1))
        if shuffle:
            random.shuffle(r)

        indices_subset = r[:self.no_of_test_images]

        return indices_subset

    def show_label_distribution(self):
        images = []
        if self.subset:

            for i, (image, label, path) in tqdm(enumerate(torch.utils.data.DataLoader(self.dataset, sampler=SubsetRandomSampler(self.index), batch_size=64, num_workers=4))):
                images += list(path)
        else:
            images = glob(f"{self.dataset_path}/*/*.jpg")

        classes_arr = [path.split("/")[-2] for path in images]
        values = [v for k, v in Counter(classes_arr).items()]
        classes = [k for k, v in Counter(classes_arr).items()]
        # print([f"{k}: {v}" for k,v in Counter(classes_arr).items()])
        print("-"*50)
        fig, ax = plt.subplots(figsize=(12, 5))
        total_images_count = sum(values)
        for k, v in Counter(classes_arr).items():
            print(f"{k}: {v} ({round(float(v/total_images_count)*100, 2)}%)")
            plt.bar(k, v)
        print("-"*50)
        plt.xticks(np.arange(len(classes)), (classes))
        plt.show()
