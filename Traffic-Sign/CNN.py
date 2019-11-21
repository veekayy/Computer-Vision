'''
Created on 13-Nov-2019

@author: vivek
'''

import os
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Conv2D, Activation, BatchNormalization, MaxPooling2D, Flatten
from keras.layers import Dropout, Dense


class TrafficSign_Net(object):
    
    @staticmethod
    
    def cnn_build(height, width, depth, n_classes):
        
        input_size= (height, width, depth)
        change_dim= -1
        #CNN1
        model = Sequential()
        
        model.add(Conv2D(8, (5,5), padding= "same", input_shape= input_size))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=change_dim))
        model.add(MaxPooling2D(pool_size=(2,2)))
        
        #CNN2
        
        model.add(Conv2D(16, (3,3), padding= "same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=change_dim))
        model.add(Conv2D(16,(3,3), padding= "same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=change_dim))
        model.add(MaxPooling2D(pool_size=(2,2)))
        
        #CNN3
        model.add(Conv2D(32, (3,3), padding= "same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=change_dim))
        model.add(Conv2D(32, (3,3), padding= "same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=change_dim))
        model.add(MaxPooling2D(pool_size=(2,2)))
        
        #FC1
        model.add(Flatten())
        model.add(Dense(128))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(.5))
        
        #FC2
        model.add(Dense(128))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(.5))
        
        #Output_layer
        model.add(Dense(n_classes))
        model.add(Activation("softmax"))
        
        
        return model
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        