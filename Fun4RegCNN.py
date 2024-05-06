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


#%% ===========================================================================
def evaluateRegModel(model,x_test,y_test):
    mse, mae = model.evaluate(x_test, y_test, batch_size = None, verbose=0)
    print('Loss as Mean squared error from neural net: ', mse)
    print('Mean absolute error from neural net: ', mae)
    predictions = model.predict(x_test).flatten()
    return predictions

    
#%% Function to plot predictions

def plotPredictionsReg(predictions,y_test,plot):
    pearson=scipy.stats.pearsonr(predictions,y_test)
    if plot :
        plt.figure()
        plt.scatter(predictions,y_test)
        
        # print(pearson)
        lims=[min(y_test)-1,max(y_test)+1]
        plt.plot(lims,lims)
        plt.xlabel('predicted')
        plt.ylabel('ture values')
        plt.xlim(lims)
        plt.ylim(lims)
        plt.show()
    return pearson[0]

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

def ResNet_Sant(input_shape):
    inputs = tf.keras.Input(shape=input_shape)
    base_model = tf.keras.applications.ResNet50(include_top=False, input_tensor=inputs, pooling='avg', weights=None)    
    x = Conv2D(64, (5, 5), strides = (1, 1), activation='relu')(inputs)
    x = BatchNormalization(axis = 3, name = 'bn0')(x)
    x = Activation('relu')(x)
    x = AveragePooling2D((2, 2))(base_model, training=False)
    outputs = tf.keras.layers.Dense(1, activation='linear')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


#%% Modelos Diego


#%% Modelos Carlos