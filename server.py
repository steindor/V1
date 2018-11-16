#!/usr/bin/python3
import sys
import os
import random
import collections
from pathlib import Path
from flask import Flask, render_template, request
from glob import glob
import pandas as pd
import json

app = Flask(__name__)


BBOX_FOLDER = "webapp/annotation_data/"
FILENAME = 'bbox_isic.csv'

TRAINING_IMAGES_PATH = "static/images/"
TRAINING_IMAGES = glob(f"{TRAINING_IMAGES_PATH}*.jpg")

def row_exists(img_data, column, df):
    img_names = df[column].tolist()
    return (img_data[column][:-4] in img_names)

def save_img_data(img_data):
    folder_path = Path(BBOX_FOLDER)
    
    if not os.path.exists(BBOX_FOLDER):
        folder_path.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(img_data, index=[img_data['img_name']], columns=['img_name','top','left','width','height'])
        df.columns.name = "img_name"
        df.to_csv(f"{BBOX_FOLDER}/{FILENAME}", index=False)
    else:
        df = pd.read_csv(f"{BBOX_FOLDER}/{FILENAME}")
        df2 = pd.DataFrame(img_data, index=[0], columns=['img_name','top', 'left', 'width', 'height'])

        if row_exists(img_data, "img_name", df):
            print("For now, do nothing. Later, update row?")
        else:
            frames = [df,df2]
            df = pd.concat(frames, ignore_index=True)
            df.to_csv(f"{BBOX_FOLDER}/{FILENAME}", index=False)
            print("Added new image to csv")
    
def get_new_photo():
    df = pd.read_csv(f"{BBOX_FOLDER}/{FILENAME}")
    next_im = random.choice(TRAINING_IMAGES)
    im_name = next_im.split("/")[-1]
    imgs_done = df['img_name'].tolist()
    if im_name in imgs_done:
        print("The name is already annotated, fetching a new one")
        get_new_photo()
    else:
        return next_im

""" 
    Removes duplicate images if they are saved by mistake in the CSV file
"""

def remove_duplicates():
    df = pd.read_csv(f"{BBOX_FOLDER}/{FILENAME}")
    imgs_done = df['img_name'].tolist()
    duplicates = [img_name for img_name,count in collections.Counter(imgs_done).items() if count > 1]
    for img in duplicates:
        print("Found duplicate for image: {}, cleaning up by deleting 1 of 2 images".format(img))
        images_df = (df.loc[df['img_name'] == img])
        indexes = images_df.index.tolist()
        print("Dropping image: ", images_df['img_name'].tolist()[1])
        print(indexes)
        df = df.drop(indexes[1])
    df.to_csv(f"{BBOX_FOLDER}/{FILENAME}", index=False)
    
    

@app.route('/')
def index():
    image = get_new_photo()
    return render_template("index.html", image=image)

@app.route("/save_bbox", methods=['POST'])
def save_bbox():
    img_data = json.loads(request.form['img_data'])

    for img in img_data:        
        img_path = img['path']
        img_name = img_path.split("/")[-1][:-4]
        img['img_name'] = img_name
        save_img_data(img)
    print("Saved {} bounding boxes".format(len(img_data)))
    photo = get_new_photo()
    return photo


# get_duplicates()

if __name__ == "__main__":
    app.config['ENV'] = 'development'
    app.config['DEBUG'] = True
    app.config['TESTING'] = True
    app.run(host='0.0.0.0', port='5000')
    
