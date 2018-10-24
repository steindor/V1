import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from IPython.core.debugger import set_trace
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms.functional as F
from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize
from PIL import Image
from glob import glob
from tqdm import tqdm
from collections import Counter
import modules.custom_transforms as custom_transforms
import matplotlib.pyplot as plt
import numpy as np
import random


# TODO: Implementera validation fold i x split

def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:, i, :, :].mean()
            std[i] += inputs[:, i, :, :].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std


def binarize_image(img_path, threshold):
    """Binarize an image."""
    image_file = Image.open(img_path)
    image = image_file.convert('L')  # convert image to monochrome
    image = image.resize((388, 388), Image.ANTIALIAS)
    image = np.array(image)
    image = binarize_array(image, threshold)
    return image


def binarize_array(numpy_array, threshold=200):
    """Binarize a numpy array."""
    for i in range(len(numpy_array)):
        for j in range(len(numpy_array[0])):
            if numpy_array[i][j] > threshold:
                numpy_array[i][j] = 255
            else:
                numpy_array[i][j] = 0
    return numpy_array

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

    def __init__(self, batch_size, dataset_path, img_size=(224, 224), shuffle=False, test=False, augment=False, subset=False, subset_percentage=None, normal_distribution=True, distribution_dict=None):
        self.batch_size = batch_size
        self.dataset_path = dataset_path
        self.img_size = img_size
        self.test = test
        self.shuffle = shuffle
        self.augment = augment
        self.subset = subset
        self.subset_percentage = subset_percentage
        self.transforms_list = [transform.split(")")[0].strip()+")" for transform in str(self.transform()).split("\n")[1:-3]]

        self.dataset = ImageFolderWithPaths(self.dataset_path, transform=self.transform())
        self.image_count = int(np.floor(subset_percentage * len(self)))

        if subset_percentage and not subset:
            raise ValueError("Subset is set to false but the percentage is set!")
        elif subset and not subset_percentage:
            raise ValueError("Subset is set to true but the percentage is not set. Choose a value for subset_percentage!")
        
        if subset:
            # Get a subset of dataset
            
            if normal_distribution and not distribution_dict:
                
                # ISIC 2018 test set distribution
                distribution_dict = {
                    "melanocytic_nevi": 0.68,
                    "melanoma": 0.11,
                    "dermatofibroma": 0.01,
                    "actinic_keratosis": 0.03,
                    "BCC": 0.05,
                    "benign_keratosis": 0.11,
                    "vascular_lesion": 0.01,
                }

                self.index = self.get_subsample_indices(shuffle=shuffle, percentage=subset_percentage, distribution_dict=distribution_dict)

                image_count = [round(self.image_count*percentage) for class_type,percentage in distribution_dict.items()]
               


            elif not normal_distribution and distribution_dict:
                print("Getting data like defined in distribution dict - distribution dict still missing")
                # distr test set: [{'MEL': 11.0}, {'NV': 67.0}, {'BCC': 5.0}, {'AKIEC': 3.0}, {'BKL': 11.0}, {'DF': 1.0}, {'VASC': 1.0}]
        else:
            print("Getting all data")

        print("*"*25)
        print(f"Total no of images: {len(self)} (x no of images)")
        if self.subset:
            print("Getting a subset of the whole dataset")
        print(f"Percentage: {self.subset_percentage}")
        print(f"Augmentation is set to: {self.augment}")
        if normal_distribution:
            print("Distribution of dataset: ISIC 2018")
        print(f"Distribution of dataset: ")
        print(f"Shuffle is set to: {self.shuffle}")
        print(f"Img size is {self.img_size[0]}x{self.img_size[1]}")
        print("*"*25)

        # train_t = transforms.Compose([transforms.ToTensor()])
        # train_set = ImageFolderWithPaths(self.dataset_path, transform=train_t)


        # breyta i generator, iterera yfir og saekja bara myndina
        # muna ad resiza fyrst i 224,224 til ad fa rett

        # print(train_set.train_data.shape)
        # print(train_set.train_data.mean(axis=(0,1,2)/255))
        # print(train_set.train_data.std(axis=(0,1,2)/255))
       

    def __getitem__(self, index):
        data, labels, path = self.dataset[index]
        return data, labels, index, path

    def __len__(self):
        return len(self.dataset)


# TODO: Implementera transforms

# data_dir = "data/"
# input_shape = 299
# batch_size = 32
# mean = [0.5, 0.5, 0.5]
# std = [0.5, 0.5, 0.5]
# scale = 360
# input_shape = 299
# use_parallel = True
# use_gpu = True
# epochs = 100

    # data_transforms = {
#     'train': transforms.Compose([
#         transforms.Resize(scale),
#         transforms.RandomResizedCrop(input_shape),
#         transforms.RandomHorizontalFlip(),
#         transforms.RandomVerticalFlip(),
#         transforms.RandomRotation(degrees=90),
#         transforms.ToTensor(),
#         transforms.Normalize(mean, std)]),
#     'val': transforms.Compose([
#         transforms.Resize(scale),
#         transforms.CenterCrop(input_shape),
#         transforms.ToTensor(),
#         transforms.Normalize(mean, std)]), }


# TODO: 
# Resize images - the short side is 1.25x larger than the input size - next a random square crop with the sice in
# [0.8,1.0] of the resized image is taken and resized to the desired input size of the model
# Next is a random horizontal flip
# Random rotation of [0,90,180,270]
# Augment brightness, saturation and contrast by random factor in the range of [0.9,1.1]

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

        # transforms.Resize(299),
        # transforms.RandomRotation(degrees=90),
        # transforms.CenterCrop(224),
        # transforms.ToTensor()

        # ratio = float(img.size[1] / img.size[0])
            # # 1.25x the input size?
            # We resize all images so the short side is 1.25x larger then the input size:
            #1 check both sides and find the shorter side
            #2 calculate the ratio between sides
            #3 make the shorter side 1.25x larger then the input size
            #4 calculate the length of the longer size
            # resize
            # random square crop [0.8,1.0] of the image
            # resize that to input img size
            # random horizontal flips
            # random rotations of [0,90,180,270]
            # augment brightness, saturation and contrast by random factor [0.9,1.1]
            #implement and check lr rate on augment vs non augment


        if self.augment:
            train_transform = transforms.Compose([
                transforms.RandomRotation((90,270)),
                custom_transforms.RatioCrop(ratio=0.9, random=True),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                transforms.CenterCrop(224),
                # transforms.Resize(size=self.img_size),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        else:
            train_transform = transforms.Compose([
                custom_transforms.ChangeBrightness(1.2),
                transforms.Resize(size=self.img_size),
                transforms.ToTensor(),
                normalize,
            ])
        
        if self.test:
            return test_transform
        else:
            return train_transform

    def get_subsample_indices(self, percentage, shuffle=None, distribution_dict=None):
        self.subsample = True
        images = {}

        for idx, (path,class_no) in enumerate(self.dataset.imgs):
            class_type = path.split("/")[-2]
            
            if not class_type in images.keys():
                images[class_type] = []
                images[class_type].append(idx)
            else:
                images[class_type].append(idx)  

        if shuffle:
            for class_type in images.keys():
                random.shuffle(images[class_type])

        indices = []

        for class_type,percentage in distribution_dict.items():
            no_of_images = round(percentage*self.image_count)
            indices.extend(images[class_type][:no_of_images])

        self.img_indices = indices

        return indices

    def show_label_distribution(self):
        images = []
        if self.subset:

            for image, label, path in (torch.utils.data.DataLoader(self.dataset, sampler=SubsetRandomSampler(self.index), batch_size=self.batch_size, num_workers=4)):
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


class FeaturesDataset(Dataset):

    def __init__(self, featlst1, featlst2, labellst):
        super(FeaturesDataset, self).__init__()
        self.featlst1 = featlst1
        self.featlst2 = featlst2
        self.labellst = labellst

    def __getitem__(self, index):
        return (self.featlst1[index], self.featlst2[index], self.labellst[index])

    def __len__(self):
        return len(self.labellst)


class SegDataset(Dataset):
    def __init__(self, root_folder, input_img_resize=(572, 572), output_img_resize=(388, 388), train_transform=False):
        """
            A dataset loader taking images paths as argument and return
            as them as tensors from getitem()
            Args:
                root_folder:
                    root_folder/train/images
                    root_folder/train/masks
                    root_folder/test/images
                    root_folder/test/masks
                    
                input_img_resize (tuple): Tuple containing the new size of the input images
                output_img_resize (tuple): Tuple containing the new size of the output images
        """
        self.root_folder = root_folder
        self.images = glob(f"{root_folder}/images/*.jpg")
        self.masks = glob(f"{root_folder}/masks/*.png")
        self.input_img_resize = input_img_resize
        self.output_img_resize = output_img_resize

    def __getitem__(self, index):
        """
            Args:
                index (int): Index
            Returns:
                tuple: (image, target) where target is class_index of the target class.
        """

        img_path = self.images[index]
        img_name = img_path.split("/")[-1][:-4]
        mask_name = f"{img_name}_segmentation.png"

        img = Image.open(img_path)
        mask = Image.open(f"{self.root_folder}/masks/{mask_name}")



        if train_transform:
            cust_transforms = custom_transforms.Compose([
                custom_transforms.RandomHorizontalFlip(),
                custom_transforms.RandomVerticalFlip(),
                custom_transforms.RandomRotate(270),
                custom_transforms.ColorJitter(
                    brightness=0.2, saturation=0.2, hue=0.2, contrast=0.1
                )
            ])

            img, mask = cust_transforms(img, mask)


        img_transform = transforms.Compose([
            transforms.Resize(self.input_img_resize),
            transforms.ToTensor(),
        ])

        mask_transform = transforms.Compose([
            transforms.Resize(self.output_img_resize),
            transforms.ToTensor(),
        ])

        img = img_transform(img)
        mask = mask_transform(mask)

        return img, mask

    def __len__(self):
        assert len(self.images) == len(self.masks)
        return len(self.images)
