'''
Created on 14-Nov-2019

@author: vivek
'''

import os
import argparse
from keras.models import load_model
from imutils import paths
import path
from skimage.io import imread
from skimage import transform, exposure
import random
import numpy as np


args= argparse.ArgumentParser()
args.add_argument("-d", "--dataset", required= True, help= "Path to test dataset")
args.add_argument("-m", "--model", required= True, help= "path to saved model")

args= vars(args.parse_args())


print("INFO Loading the pre-trained model")
model= load_model(args['model'])

image_paths= list(paths.list_images(args['dataset']))
random.shuffle(image_paths)

print("Making predictions")
for i, image_path in enumerate(image_paths):
    
    image= imread(image_path)
    image= transform.resize(image, (32,32))
    image= exposure.equalize_adapthist(image, clip_limit=.1)
    image= image/255.0
    image= np.expand_dims(image, axis=0)
    
    prediction= model.predict(image)
    test_label= prediction.argmax(axis= 1)[0]
    
    
print("Done")
    
    
    
    
    
    
    