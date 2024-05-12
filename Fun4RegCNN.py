#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  3 20:09:19 2024

@author: isaac
"""

import tensorflow as tf
import scipy 
import numpy as np
import pandas as pd
import seaborn as sns
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso, LinearRegression
from tensorflow import keras
from tensorflow.keras.metrics import Accuracy, Precision, Recall
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
# Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
# Memory growth must be set before GPUs have been initialized
        print(e)

import os
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
from tensorflow.keras import models
from tensorflow.keras.layers import Input, Dense, Activation, BatchNormalization, Flatten, Conv2D,Conv1D, concatenate
from tensorflow.keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalAveragePooling2D, MaxPooling1D
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras import optimizers
from tensorflow.keras import losses
from tensorflow.keras import Model
from sklearn import metrics as skmetrics
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from tensorflow.keras.applications import ResNet50, VGG16
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import ResNet50
from my_image_tools import rgb2gray

#%% ===========================================================================
def evaluateRegModel(model,x_test,y_test,verbose=None):
    mse, mae = model.evaluate(x_test, y_test, batch_size = None, verbose=0)
    if verbose:
        print('Loss as Mean squared error from neural net: ', mse)
        print('Mean absolute error from neural net: ', mae)
    predictions = model.predict(x_test).flatten()
    return predictions

    
#%% Function to plot predictions

def plotPredictionsReg(predictions,y_test,plot, ax=None):
    print(predictions[0:5])
    ## pearson=scipy.stats.pearsonr(predictions,y_test)
    pearson=scipy.stats.pearsonr(predictions,y_test[:,0])
    if plot :
        if ax == None:
            fig,ax=plt.subplots()
        ax.scatter(predictions,y_test)
        
        # print(pearson)
        lims=[min(y_test)-1,max(y_test)+1]
        ax.plot(lims,lims)
        ax.set_xlabel('predicted')
        ax.set_ylabel('true values')
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        plt.show()
    return pearson[0]


#%% 1D to 3D by repeating the matrix
def gray_to_rgb(img):
    gray_img = np.expand_dims(rgb2gray(img),axis=-1)
    return np.repeat(gray_img, 3, 2)

#%% Lasso
def Lasso_FC(tt_split,coeffs,ax):
    x_train, x_test,y_train, y_test = tt_split
    model = Lasso(alpha=.02,max_iter=10000)
    model.fit(x_train, y_train)
    pred_Lasso=model.predict(x_test)
    lassoPred=plotPredictionsReg(pred_Lasso,y_test,True,ax=ax)
    if coeffs:
        return model.coef_, model.intercept_, lassoPred
    return lassoPred
#%% Modelos Santiago

def CNN_Sant(input_shape):
    inputs = tf.keras.Input(shape=input_shape)
    x = Conv2D(64, (5, 5), strides = (1, 1), activation='relu')(inputs)
    x = BatchNormalization(axis = 3, name = 'bn0')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = BatchNormalization(axis = 3, name = 'bn1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(256, (3, 3), activation='relu')(x)
    # x = BatchNormalization(axis = 3, name = 'bn2')(x)
    # x = Activation('relu')(x)
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = Dense(256, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    x = Dense(32, activation='relu')(x)
    outputs = Dense(1, activation='linear')(x)  # Linear activation for regression
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

def ResNet_Sant(input_shape, freeze=True, pretrained=True):
     # inputs = tf.keras.Input(shape=input_shape)
     
     base_model = ResNet50(weights='imagenet' if pretrained else None, include_top=False, input_shape=input_shape)
    
     # Freeze layers if specified
     if freeze:
         for layer in base_model.layers:
             layer.trainable = False
        
     # Add custom classification head
     x = GlobalAveragePooling2D()(base_model.output) 
     x = Dense(512, activation='relu')(x)
     x = Dropout(.3)(x)
     # x = Flatten()(base_model.output) 
     outputs = tf.keras.layers.Dense(1, activation='linear')(x)
     model = Model(inputs=base_model.input, outputs=outputs)
    
     # Create model

     return model


def VGG16_Sant(input_shape, freeze=True, pretrained=True):
     # inputs = tf.keras.Input(shape=input_shape)
     
     base_model = VGG16(weights='imagenet' if pretrained else None, include_top=False, input_shape=input_shape)
    
     # Freeze layers if specified
     if freeze:
         for layer in base_model.layers:
             layer.trainable = False
        
     # Add custom classification head
     x = GlobalAveragePooling2D()(base_model.output) 
     x = Dense(512, activation='relu')(x)
     x = Dropout(.3)(x)
     # x = Flatten()(base_model.output) 
     outputs = tf.keras.layers.Dense(1, activation='linear')(x)
     model = Model(inputs=base_model.input, outputs=outputs)
    
     # Create model

     return model

def Perceptron_PCA (input_shape):
    # print(classification)
    tf.keras.backend.clear_session()
    inputs = tf.keras.Input(shape=input_shape)
    x = Dense(512, activation='sigmoid')(inputs)
    x = Dense(256, activation='relu')(x)
    x = Dense(64, activation='tanh')(x)
    x = Dense(16, activation='selu')(x)
    x = Dense(512, activation='sigmoid')(x)
    x = Dense(16, activation='gelu')(x)
    outputs = Dense(1, activation='linear')(x)  # Linear activation for regression
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

#%% Modelos Diego
def CNN_Diego(input_shape):
    leaky_relu = LeakyReLU(alpha=0.01)
    inputs = tf.keras.Input(shape=input_shape)
    x = Conv2D(128, (7, 7), strides = (2, 2), activation='relu')(inputs)
    x = BatchNormalization(axis = 3, name = 'bn0')(x)
    x = Activation(leaky_relu)(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(256, (5, 5), activation=leaky_relu)(x)
    x = BatchNormalization(axis = 3, name = 'bn1')(x)
    x = Activation(leaky_relu)(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(512, (3, 3), activation='relu')(x)
    # x = BatchNormalization(axis = 3, name = 'bn2')(x)
    # x = Activation('relu')(x)
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = Dense(256, activation=leaky_relu)(x)
    x = Dense(128, activation=leaky_relu)(x)
    x = Dense(64, activation='relu')(x)
    x = Dense(32, activation='sigmoid')(x)
    outputs = Dense(1, activation='linear')(x)  # Linear activation for regression
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

def Transf_Diego(input_shape):
    inputs = tf.keras.Input(shape=input_shape)
    base_model = tf.keras.applications.EfficientNetV2S(include_top=False, input_tensor=inputs, pooling='avg', weights=None)
    x = tf.keras.layers.Dense(512, activation='relu')(base_model.output)
    outputs = tf.keras.layers.Dense(1, activation='linear')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


#%% Modelos Carlos
def CNN_Carl(input_shape):
    leaky_relu = LeakyReLU(alpha=0.01)
    inputs = tf.keras.Input(shape=input_shape)
    x = Conv2D(128, (7, 7), strides = (2, 2), activation=leaky_relu)(inputs)
    x = BatchNormalization(axis = 3, name = 'bn0')(x)
    x = Activation(leaky_relu)(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(256, (5, 5), activation=leaky_relu)(x)
    x = BatchNormalization(axis = 3, name = 'bn1')(x)
    x = Activation(leaky_relu)(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(512, (2, 2), activation='relu')(x)
    # x = BatchNormalization(axis = 3, name = 'bn2')(x)
    # x = Activation('relu')(x)
    x = Flatten()(x)
    x = Dense(512, activation=leaky_relu)(x)
    x = Dense(256, activation=leaky_relu)(x)
    x = Dense(128, activation=leaky_relu)(x)
    x = Dense(64, activation=leaky_relu)(x)
    x = Dense(32, activation=leaky_relu)(x)
    outputs = Dense(1, activation='linear')(x)  # Linear activation for regression
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

def Transf_Carl(input_shape):
    inputs = tf.keras.Input(shape=input_shape)
    base_model = tf.keras.applications.EfficientNetV2S(include_top=False, input_tensor=inputs, pooling='avg', weights=None)
    x = tf.keras.layers.Dense(512, activation='relu')(base_model.output)
    outputs = tf.keras.layers.Dense(1, activation='linear')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model
