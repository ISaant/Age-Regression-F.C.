#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  2 17:41:57 2024

@author: isaac
"""

import numpy as np
import os
os.chdir('/home/isaac/Documents/Doctorado_CIC/NewThesis/Python_Fun')
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pickle
from Fun4CNNFc import *
from read_Fc import read_Fc
from Fun4RegCNN import *
from sklearn.model_selection import train_test_split

#%%

connectomes_fc = []
ROIs = []
scores = []

#%%  Directories
current_path = os.getcwd()
parentPath = os.path.abspath(os.path.join(current_path, '../../'))
path2fc = parentPath+'/NewThesis_db_DK/camcan_AEC_ortho_AnteroPosterior'
path2demo = parentPath+'/NewThesis_db_DK/camcan_demographics/'
FcFile = np.sort(os.listdir(path2fc))
demoFile = np.sort(os.listdir(path2demo))

#%% Find nan values in the score dataframe
# import Fun4newThesis
# reload(Fun4newThesis)

print('Loading the Demographics and score table...')
with open(current_path+'/scoreDf_spanish.pickle', 'rb') as f:
    scoreDf = pickle.load(f)

#lets just keep age for now:
# scoreDf.drop(columns=['Acer','BentonFaces','Cattell','EmotionRecog','Hotel','Ppp','Synsem','VSTM'],inplace=True)
scoreDf.drop(columns=['BentonFaces','ReconocimientoEmociones', 'ForceMatch', 'Hotel', 'Ppp', 'Synsem',
        'VSTM'],inplace=True)
# scoreDf.drop(columns=['Acer','BentonFaces','Cattell','ReconocimientoEmociones', 'ForceMatch', 'Hotel', 'Ppp', 'Synsem',
#         'VSTM'],inplace=True)
row_idx=np.unique(np.where(np.isnan(scoreDf.iloc[:,3:-1].to_numpy()))[0])#rows where there is nan
scoreDf_noNan=scoreDf.drop(row_idx).reset_index(drop=True)
scoreDf_noNan=scoreDf_noNan.drop(np.argwhere(scoreDf_noNan['ID']=='sub_CC721434')[0][0]).reset_index(drop=True)# drop beacuase there is missing connections at the the struct connectomics
PltDistDemographics(scoreDf_noNan)
edad=np.array(scoreDf_noNan['Edad']).reshape(-1,1)
subjects=scoreDf_noNan['ID']

# with open(current_path+'/scoreDf.pickle', 'rb') as f:
#     scoreDf_old = pickle.load(f)
# subjects_old=scoreDf_old['ID']
# row_idx=[np.argwhere(subjects_old == missing)[0][0] for missing in list(set(subjects_old).difference(set(subjects)))] #Esto solo funciona para las matrices que te paso jason

# sleep(1)
# plt.close('all')

#%% Read Fc
connectomes_fc, ROIs = read_Fc(FcFile,path2fc, subjects,thresholding='MST', per=10) #nt = no threshold
connectome = connectomes_fc['alpha']
connectome = (connectome/(np.max(connectome, axis=None)))[:,:,:,np.newaxis]
input_shape = connectome.shape[1:4]
x_train, x_test, y_train, y_test = train_test_split(connectome, edad, test_size=.2)


# Create the model
model = CNN(input_shape)

# Compile the model
model.compile(optimizer=Adam(lr=.001), loss='mean_squared_error', metrics=['mae'])

# Train the model
history = model.fit(x_train, y_train, epochs=30, batch_size=16, validation_split=0.2)

# Evaluate the model
predictions = evaluateRegModel(model,x_test,y_test)
acc = plotPredictionsReg(predictions,y_test,True)
print (acc)




