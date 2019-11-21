'''
Created on 14-Nov-2019

@author: vivek
'''

import os
import argparse
import matplotlib
from keras.utils.np_utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report

matplotlib.use("agg")
import matplotlib.pyplot as plt
from keras.optimizers import Adam
from CNN import TrafficSign_Net
from skimage.io import imread
import random
import numpy as np
from skimage import transform, exposure
from keras.utils import to_categorical

def load_data(basepath, csvpath):
    
    data= []
    labels= []
    #print(csvpath)
    rows= open(csvpath).read().strip().split("\n")[1:]
    random.shuffle(rows)
    
    for i, row in enumerate(rows):
        
        if i > 0 and i % 1000 == 0:
            print("Processed {} images".format(i))
        
        (label, image)= row.strip().split(",")[-2:]
        
        imagepath= os.path.join(basepath, image)
        img= imread(imagepath)
        img= transform.resize(img, (32,32))
        #img= exposure.equalize_adapthist(image, clip_limit=0.1)
  
        data.append(img)
        labels.append(label)
        
    data= np.array(data)
    labels= np.array(labels)
    
    return (data, labels)


args= argparse.ArgumentParser()
args.add_argument("-d", "--dataset", required= True, help= "data path")
args.add_argument("-m", "--model", required= True, help= "path to the model output")
args.add_argument("-p", "--plot", type= str, default= "plot.png", help= "path to the output plot")

args= vars(args.parse_args())


#model parameters
n_epochs= 30
bs= 64
lr= .001
(h,w,d)= (32,32,3)

print("INFO loading the data")
#basepath= os.curdir()
filepath= os.path.join("Downloads/eclipse/home/vivek/Documents/Traffic_sign","gtsrb-german-traffic-sign")

trainpath= os.path.join(args['dataset'], "Train.csv")
testpath= os.path.join(args['dataset'], "Test.csv")

#print(trainpath, testpath)
train_X, train_y= load_data(args['dataset'],trainpath)
test_X, test_y= load_data(args['dataset'],testpath)

train_X= train_X.astype("float")/255.0
test_X= test_X.astype("float")/255.0

train_y= to_categorical(train_y)
test_y= to_categorical(test_y)
print(train_y)

n_labels= len(np.unique(train_y))
class_total= np.sum(train_y, axis= 0)
class_weight= class_total.max()/class_total

aug = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.15,
    horizontal_flip=False,
    vertical_flip=False,
    fill_mode="nearest")

#compiling model
print("INFO compiling model")
opt= Adam(lr= lr, decay= lr/(n_epochs*.5))
model= TrafficSign_Net.cnn_build(h,w,d, n_labels)
model.compile(loss= "categorical_crossentropy", optimizer=opt, metrics=['accuracy'])

#Trainig model
print("Training the network")
H = model.fit_generator(
    aug.flow(train_X, train_y, batch_size=bs),
    validation_data=(test_X, test_y),
    steps_per_epoch=train_X.shape[0] // bs,
    epochs=n_epochs,
    class_weight=class_weight,
    verbose=1)

print("Evaluating the network")

pred= model.predict(test_X, batch_size= bs)

print("---classfication report---")
print(classification_report(test_y.argmax(axis=1), pred.argmax(axis= 1)))

print("INFO serializing network to '{}'".format(args['model']))


print("Plotting the training and validation accrucy")
N = np.arange(0, n_epochs)
plt.style.use("ggplot")
plt.figure()
plt.plot(N, H.history["loss"], label="train_loss")
plt.plot(N, H.history["val_loss"], label="val_loss")
plt.plot(N, H.history["train_acc"], label="train_acc")
plt.plot(N, H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])





