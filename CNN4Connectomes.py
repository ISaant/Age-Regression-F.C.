#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 21:52:31 2024

@author: isaac
"""

import matplotlib.pyplot as plt
import numpy as np
import os
from scipy import stats
import pickle
os.chdir('/home/isaac/Documents/Doctorado_CIC/NewThesis/Python_Fun')
from Generate_Features_Dataloades import Generate
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPool2D, BatchNormalization
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

# import os
# os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
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


#%%
gen = input('read (r) or generate(g)')
if gen == 'g':
    psd2use, restStatePCA, anat2use, anatPCA, DiagFc,connectomes_fc, restStatePCA_s200, anatPCA_s200, structConn, local, glob, ROIs, scores, subjects = Generate(fc_per=100)
    with open(os.getcwd()+'/DataGenerated.pickle', 'wb') as f:
        pickle.dump([psd2use, restStatePCA, anat2use, anatPCA, DiagFc,connectomes_fc, restStatePCA_s200, anatPCA_s200, structConn, local, glob, ROIs, scores, subjects], f)

with open('DataGenerated.pickle','rb') as f:
    [psd2use, restStatePCA, anat2use, anatPCA, DiagFc,connectomes_fc, restStatePCA_s200, anatPCA_s200, structConn, local, glob, ROIs, scores, subjects] = pickle.load(f)
    
#%%

connectome = connectomes_fc['alpha']
# connectome = stats.zscore(connectome, axis=None)[:,:,:,np.newaxis]
connectome = (connectome/(np.max(connectome, axis=None)))[:,:,:,np.newaxis]

input_shape = connectome.shape[1:4]
# min_age = np.min(scores)
# max_age = np.max(scores)
# scores = (scores - min_age) / (max_age - min_age)
# for i in range(10):
x_train, x_test, y_train, y_test = train_test_split(connectome, scores, test_size=.2)

# def CNN(input_shape):
#     inputs = tf.keras.Input(shape=input_shape)
#     x = Conv2D(32, (5, 5), strides = (1, 1), activation='relu')(inputs)
#     x = BatchNormalization(axis = 3, name = 'bn0')(x)
#     x = Activation('relu')(x)
#     x = MaxPooling2D((2, 2))(x)
#     x = Conv2D(, (5, 5), activation='relu')(x)
#     x = BatchNormalization(axis = 3, name = 'bn1')(x)
#     x = Activation('relu')(x)
#     x = MaxPooling2D((2, 2))(x)
#     x = Conv2D(64, (3, 3), activation='relu')(x)
#     x = BatchNormalization(axis = 3, name = 'bn2')(x)
#     x = Activation('relu')(x)
#     x = MaxPooling2D((2, 2))(x)
#     x = Flatten()(x)
#     x = Dense(2048, activation='sigmoid')(x)
#     x = Dense(1024, activation='relu')(x)
#     x = Dense(512, activation='relu')(x)
#     x = Dense(32, activation='relu')(x)
#     x = Dense(512, activation='relu')(x)
#     x = Dense(16, activation='relu')(x)
#     outputs = tf.keras.layers.Dense(1, activation='linear')(x)  # Linear activation for regression
#     model = tf.keras.Model(inputs=inputs, outputs=outputs)
#     return model

def CNN(input_shape):
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

def ResNet(input_shape):
    inputs = tf.keras.Input(shape=input_shape)
    base_model = tf.keras.applications.ResNet50(include_top=False, input_tensor=inputs, pooling='avg', weights=None)
    x = tf.keras.layers.Dense(512, activation='relu')(base_model.output)
    outputs = tf.keras.layers.Dense(1, activation='linear')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model
# Create the model
# model = CNN(input_shape)
model = CNN(input_shape)

# Compile the model
model.compile(optimizer=Adam(lr=.001), loss='mean_squared_error', metrics=['mae'])

# Train the model
history = model.fit(x_train, y_train, epochs=300, batch_size=16, validation_split=0.2)

# Evaluate the model (optional)
loss, mae = model.evaluate(x_test, y_test)
print("Test Loss:", loss)
print("Test MAE:", mae)