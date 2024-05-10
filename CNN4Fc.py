#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  2 17:41:57 2024

@author: isaac
"""

import numpy as np
import os
os.chdir('/home/isaac/Documents/Doctorado_CIC/NewThesis/FunctionalConnectivityRepo')
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pickle
from Fun4CNNFc import *
from read_Fc import read_Fc
from read_Sc import read_Sc
from Fun4RegCNN import *
from sklearn.model_selection import train_test_split
from tqdm import tqdm 

#%%

connectomes_fc = []
ROIs = []
scores = []

#%%  Directories
current_path = os.getcwd()
parentPath = os.path.abspath(os.path.join(current_path, '../../'))
path2fc = parentPath+'/NewThesis_db_DK/camcan_AEC_ortho_AnteroPosterior'
# path2fc = parentPath+'/NewThesis_db_DK/camcan_AEC_ortho_YEO' #no pelen esto, es otra forma de organizar las matrices segun YEO (sí, es chino)
path2sc = parentPath+'/NewThesis_db_s200/msmtconnectome'
path2demo = parentPath+'/NewThesis_db_DK/camcan_demographics/'
FcFile = np.sort(os.listdir(path2fc))
ScFile = np.sort(os.listdir(path2sc))
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
connectomes_fc, ROIs = read_Fc(FcFile,path2fc, subjects,thresholding='MST', per=100) #nt = no threshold
for key in connectomes_fc.keys():
    connectomes_fc[key] = np.expand_dims(connectomes_fc[key],-1) # we need to expand the dimensions to be able to concatenate the bands
#%% Testing Lasso for ground truth
'''Aqui se hace una regresion linear usando Lasso. Los valores estan normalizados con respecto al
máximo global de cada banda. No se puede normalizar cada matriz de manera independiente porque 
perderiamos la "intensidad" relacionada con la diferencia de edades.

Lasso es, como cualquier regresion lineal, determinista. Esto quiere decir que siempre dara el mismo 
resultado. La ventaja de las es que si una variable no es explicativa, su peso asociado es CERO.
La pregunta es ¿Que tan estable es la seleccion de las variables explicativas si aleatorizamos los 
datos?'''
keys= list(connectomes_fc.keys()) #extraemos las etiquetas del diccionario para iterar sobre de ellas
LassoResults=pd.DataFrame( np.zeros((10,6)),columns=keys) #generamos un dataframe vacio
coeffsIntercepts=[] #list vacia para capturar los coeficientes e intersección de la regresión
coeffsMasks=[] # Los coeficientes son asociados a los valores de l
triu_index=np.triu_indices(68,k=1) #indices del triángulo superior de la matriz
for i,key in enumerate(keys):
    upper_triangle=connectomes_fc[key][:,triu_index[0],triu_index[1],0] #seleccionamos y vectorizamos el triangulo superior del las matrices
    upper_triangle /= np.max(upper_triangle,axis=None) #normalizamos
    Coeffs=[] #listas vacias para capturar los coeficientes
    Inter=[] # = las intersecciones
    Acc=[] # = el accuracy
    fig,ax = plt.subplots()
    for j in range(10):
        tt_split = train_test_split(upper_triangle, edad, test_size=.2, random_state=j) #separamos en train/test
        coeffs,inter, acc=Lasso_FC(tt_split,True,ax) #entrenamos el modelo y lo evaluemos. La función regresa los parametros de la recta y el acc
        Coeffs.append(coeffs) # adjunta
        Inter.extend(inter) #adjunta
        Acc.extend(acc) #adjunta
    plt.title(key)
    mask=Coeff_Heatmaps(Coeffs) # Esta funcion regresa el promedio y el std de 
    #los coeficientes en su respectivo espacio en la matriz, para ver que tan estables son (son rete-estables)
    #Recordemos que antes vectorizamos los #pesos de la matriz de conectividad,
    #eso quiere decir que lasso agregará un "peso" a cada valor de conectividad.
    #Este peso puede ser mapeado nuevamente a su posicion en la matriz de conectividad.
    #Se llama "máscara" porque se sobreprondra sobre la matriz de conectividad para 
    #seleccionar los valores de connectividad que tengan un peso"
    mask[np.tril_indices(68,k=0)]=0 #seleccionamos el triangulo inferior (podría tomarse el superior, es igual)
    coeffsMasks.append(mask) #guardamos
    plt.suptitle(key) #ploteamos
    LassoResults[key]=np.array(Acc)
DataFrameMelted=pd.melt(LassoResults) #"derretimos" el dataframe. Sugiero leer, esta muy largo explicar xD
plt.figure()
sns.boxplot(x='variable',y='value',data=DataFrameMelted, palette='mako')

#%% Lets analize the coeffs. what if we use them as a mask to extract the features, PCA them and use MLP?
'''
Aquí primero seleccionamos los indices de los valores que no son ceros en la máscara, para luego
seleccionar esos mismos valores en las matrices de conectividad y vectorizarlos.

Para saber que tan explicativos son y si es posible reducir el número de dimensiones, se usó PCA.

Ya que delta, alfa y beta son las bandas que mejor predicen (segun Lasso), seleccionamos el número de PCAs
que de cada banda que explicaran almenos el 90%

Tendiendo eso se entreno un MLP.'''

#PCA
plt.figure()
for i,key in enumerate(keys):
    nonzero_index=np.nonzero(coeffsMasks[i]) #Seleccion de coeficientes != cero
    alpha_weigths = connectomes_fc[key][:,nonzero_index[0],nonzero_index[1],0] #mapeamos a la matriz de conectividad y vectorizamos
    pca_df, pca2use, prop_varianza_acum = myPCA(alpha_weigths, False, 120) #PCA
    plt.plot(prop_varianza_acum[:400],label=key) #graficamos el %de varianza explicada acumulada
plt.legend()

#seleccionamos el # de PCAs de cada banda (100,120,160)
nonzero_index=np.nonzero(coeffsMasks[0])
alpha_weigths = connectomes_fc['delta'][:,nonzero_index[0],nonzero_index[1],0]
pca_df, pca_d, prop_varianza_acum = myPCA(alpha_weigths, False, 100)
nonzero_index=np.nonzero(coeffsMasks[1])
alpha_weigths = connectomes_fc['alpha'][:,nonzero_index[0],nonzero_index[1],0]
pca_df, pca_a, prop_varianza_acum = myPCA(alpha_weigths, False, 120)
nonzero_index=np.nonzero(coeffsMasks[3])
alpha_weigths = connectomes_fc['beta'][:,nonzero_index[0],nonzero_index[1],0]
pca_df, pca_b, prop_varianza_acum = myPCA(alpha_weigths, False, 160)

features= np.concatenate((np.array(pca_d),np.array(pca_a),np.array(pca_b)),axis=1) #concatenamos los pca
x_train, x_test, y_train,y_test=train_test_split(features,edad,test_size=.2,random_state=726) #separamos en train-test
input_shape=x_train.shape[1]
# anatPCA[:, :, roi] = np.array(pca2use)

#Yaaaaa estoy ya lo conocen
model = Perceptron_PCA(input_shape)

# Compile the model
model.compile(optimizer=Adam(lr=.001), loss='mean_squared_error', metrics=['mae'])

# Train the model
history = model.fit(x_train, y_train, epochs=100, batch_size=32, validation_split=0.1, verbose=1)

# Evaluate the model
predictions = evaluateRegModel(model,x_test,y_test,verbose=True)
acc = plotPredictionsReg(predictions,y_test,True)
print ('Acc= ',acc[0])


#%%

connectome_rgb=Bands2rgb(['delta','alpha','beta'], connectomes_fc, Norm=True)
input_shape = connectome_rgb.shape[1:4]
x_train, x_test, y_train, y_test = train_test_split(connectome_rgb, edad, test_size=.2)

#%% training and testing

# Create the model
model = CNN_Sant(input_shape)

# Compile the model
model.compile(optimizer=Adam(lr=.001), loss='mean_squared_error', metrics=['mae'])

# Train the model
history = model.fit(x_train, y_train, epochs=100, batch_size=16, validation_split=0.2)

# Evaluate the model
predictions = evaluateRegModel(model,x_test,y_test)
acc = plotPredictionsReg(predictions,y_test,True)
print (acc)




