import random
import shutil

""" 
    Divides dataset from {images path} and {masks path} to:
        ../train/images
        ../train/masks
        ../test/images
        ../test/masks

        Args:
            ratio(float): the divide ratio (e.g. 0.2 for 80 vs 20 split)
            images_path(string): the path to the root folder containing the training images
            masks_path(string): the path to the root folder containing the masks
"""

def divide_dataset(ratio, images_path, masks_path):
    train_folder_path = f"{PATH}train/"
    test_folder_path = f"{PATH}test/"
    imgs = glob(f"{images_path}/*.jpg")
    masks = glob(f"{masks_path}/*.png")
    no_of_samples = len(imgs)
    no_train_imgs = int(ratio*no_of_samples)
    no_of_test_imgs = no_of_samples - no_train_imgs
    random.shuffle(imgs)

    train_imgs_split = imgs[:no_train_imgs]
    test_imgs_split = imgs[no_train_imgs:]

    for idx, img in enumerate(train_imgs_split):
        img_name = img.split("/")[-1][:-4]
        mask_name = f"{img_name}_segmentation.png"
        mask_path = f"{masks_path}{mask_name}"
        shutil.copy(img, f"{train_folder_path}images/{img_name}.jpg")
        shutil.copy(mask_path, f"{train_folder_path}masks/{mask_name}")

    for idx, img in enumerate(test_imgs_split):
        img_name = img.split("/")[-1][:-4]
        mask_name = f"{img_name}_segmentation.png"
        mask_path = f"{masks_path}{mask_name}"
        shutil.copy(img, f"{test_folder_path}images/{img_name}.jpg")
        shutil.copy(mask_path, f"{test_folder_path}masks/{mask_name}")
