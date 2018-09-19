
import pandas as pd
import multiprocessing
import os
import urllib.request as urllib2
import urllib
import random
from multiprocessing.dummy import Pool as ThreadsPool
from os.path import expanduser

from collections import Counter

from functools import partial

class ImageDownloader():

    def __init__(self):
        self.ISIC_ENDPOINT = 'https://isic-archive.com/api/v1/image/{}/download'
        self.OUT_PATH = 'downloaded_images/{}/{}/{}.jpg'

    def download_images(self, class_type=None):
        metadata = pd.read_csv('./image_downloader/metadata_w_diagnosis.csv')
        
        if class_type:
            metadata_t = metadata.loc[metadata['diagnosis'] == class_type]
        

        possible_classes = set(list(metadata['diagnosis']))
        print(possible_classes)

        ids = metadata['id']
        names = metadata['name']
        malignancy = metadata['benign_malignant']
        diagnosis = metadata['diagnosis']

        print(Counter(diagnosis))

        print(diagnosis[:10])        
        # partial_save = partial(download_image)
        # pool = ThreadsPool(4)
        # pool.map(partial_save, zip(ids, names, malignancy, diagnosis))
        # pool.close()
        # pool.join()

    def download_image(self, datum):
        img_id, img_name, benign_mal, diagnosis = datum
        # img = urllib2.urlopen(ISIC_ENDPOINT.format(img_id))
        # subdir = 'benign' if benign_mal == 0  else 'malignant'
        # BKL: Benign keratosis(solar lentigo / seborrheic keratosis / lichen planus like keratosis)
        # AKIEC: Actinic keratosis / Bowens disease

        train_test = random.uniform(0, 1)

        subdir = 'test' if train_test < 0.2 else 'train'

        if not os.path.exists('downloaded_images/{}'.format(subdir)):
            os.makedirs('downloaded_images/{}'.format(subdir))

        folders = ['melanoma', 'dermatofibroma', 'melanocytic_nevi', 'BCC',
                'vascular_lesion', 'benign_keratosis', 'actinic_keratosis']
        for folder in folders:
            if not os.path.exists('downloaded_images/{}/{}'.format(subdir, folder)):
                os.makedirs('downloaded_images/{}/{}'.format(subdir, folder))

        if diagnosis == 'melanoma' or diagnosis == 'dermatofibroma':
            trn_path = self.OUT_PATH.format('train', diagnosis, img_name)
            test_path = self.OUT_PATH.format('test', diagnosis, img_name)
        elif diagnosis == 'basal cell carcinoma':
            trn_path = self.OUT_PATH.format('train', 'BCC', img_name)
            test_path = self.OUT_PATH.format('test', 'BCC', img_name)
        elif diagnosis == 'nevus':
            trn_path = self.OUT_PATH.format('train', 'melanocytic_nevi', img_name)
            test_path = self.OUT_PATH.format('test', 'melanocytic_nevi', img_name)
        elif diagnosis == 'vascular lesion':
            trn_path = self.OUT_PATH.format('train', 'vascular_lesion', img_name)
            test_path = self.OUT_PATH.format('test', 'vascular_lesion', img_name)
        elif diagnosis == 'actinic keratosis':
            trn_path = self.OUT_PATH.format('train', 'actinic_keratosis', img_name)
            test_path = self.OUT_PATH.format('test', 'actinic_keratosis', img_name)
        elif diagnosis in ['seborrheic keratosis', 'lentigo']:
            trn_path = self.OUT_PATH.format('train', 'benign_keratosis', img_name)
            test_path = self.OUT_PATH.format('test', 'benign_keratosis', img_name)
        else:
            print("diagnosis not included: {}".format(diagnosis))

        try:
            image_path = trn_path if subdir is "train" else test_path
        except UnboundLocalError:
            print("UnboundLocalError, unsusre which error this is, no path?")

        # print(f"Downloaded image with diagnosis {diagnosis} to path: {image_path}")

        # kemur error i mynd nr 13.608 + 3.379 = 16.987

        # urllib.request.urlretrieve(self.ISIC_ENDPOINT.format(img_id), image_path)
