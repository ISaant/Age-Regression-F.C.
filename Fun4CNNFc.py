#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  3 19:48:08 2024

@author: isaac
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
from copy import copy
import seaborn as sns
sns.set_context("poster",font_scale=1,rc={"lines.linewidth": 3})
from tqdm import tqdm
from scipy import sparse
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn import linear_model, svm, preprocessing, cross_decomposition



#%% Gradientes de color =======================================================
def hex_to_RGB(hex):
  ''' "#FFFFFF" -> [255,255,255] '''
  # Pass 16 to the integer function for change of base
  return [int(hex[i:i+2], 16) for i in range(1,6,2)]

def RGB_to_hex(RGB):
  ''' [255,255,255] -> "#FFFFFF" '''
  # Components need to be integers for hex to make sense
  RGB = [int(x) for x in RGB]
  return "#"+"".join(["0{0:x}".format(v) if v < 16 else
            "{0:x}".format(v) for v in RGB])

def color_dict(gradient):
  ''' Takes in a list of RGB sub-lists and returns dictionary of
    colors in RGB and hex form for use in a graphing function
    defined later on '''
  return {"hex":[RGB_to_hex(RGB) for RGB in gradient],
      "r":[RGB[0] for RGB in gradient],
      "g":[RGB[1] for RGB in gradient],
      "b":[RGB[2] for RGB in gradient]}

def linear_gradient(start_hex, finish_hex, n):
  ''' returns a gradient list of (n) colors between
    two hex colors. start_hex and finish_hex
    should be the full six-digit color string,
    inlcuding the number sign ("#FFFFFF") '''
  # Starting and ending colors in RGB form
  s = hex_to_RGB(start_hex)
  f = hex_to_RGB(finish_hex)
  # Initilize a list of the output colors with the starting color
  RGB_list = [s]
  # Calcuate a color at each evenly spaced value of t from 1 to n
  for t in range(1, n):
    # Interpolate RGB vector for color at the current value of t
    curr_vector = [
      int(s[j] + (float(t)/(n-1))*(f[j]-s[j]))
      for j in range(3)
    ]
    # Add it to our list of output colors
    RGB_list.append(curr_vector)

  return color_dict(RGB_list)

#%% ===========================================================================
def PltDistDemographics(demographics):
    sns.violinplot(data=demographics, x='Sexo', y='Edad', palette='mako', inner="point")
    plt.title('Distribución de las edades por genero')
    Age = demographics['Edad'].to_numpy()
    Acer = demographics['Acer'].to_numpy()
    Cattell = demographics['Cattell'].to_numpy()
    # RoundAge = copy(Age)
    # RoundAge[RoundAge < 30] = 30
    # for i in np.arange(30, 90, 10):
    #     print(i)
    #     RoundAge[np.logical_and(RoundAge > i, RoundAge <= i + 10)] = (i + 10)
    # # RoundAge[RoundAge>80]=90
    # demographics['Intervalo'] = RoundAge
    sns.displot(data=demographics, x='Cattell', hue='Intervalo', kind='kde', fill=True)
    plt.title('Distribución de Cattell por rango de edad')
    plt.ylabel('Densidad')
    sns.displot(data=demographics, x='Acer', hue='Intervalo', kind='kde', fill=True)
    plt.title('Distribución de ACE-R por rango de edad')
    plt.ylabel('Densidad')
    plt.xlim([60, 110])
    # plt.figure()
    # sns.lmplot(x='Age', y='Cattell', data=demographics,
    #            scatter=False, scatter_kws={'alpha': 0.3}, palette='CMRmap')
    plt.figure()
    sns.residplot(data=demographics, x="Edad", y="Cattell", order=2, line_kws=dict(color="r"))
    # plt.figure()
    # sns.residplot(data=demographics, x="Edad", y="Cattell", order=2, line_kws=dict(color="r"))
    sns.relplot(data=demographics, y='Cattell', x='Edad', hue='Intervalo')
    plt.title('Regresión de Cattell con respecto a la Edad ')
    rsq, pvalue = scipy.stats.pearsonr(Age, Cattell)
    rsq_cuad, pvalue_cuad = scipy.stats.pearsonr(Age**2, Cattell**2)
    Age = Age.reshape(-1, 1)
    linReg = linear_model.LinearRegression()
    # linReg.fit(Edad, Cattell)
    # Predict data of estimated models
    # line_age = np.round(np.arange(Age.min() - 5, Age.max() + 5, .01), 2)[:, np.newaxis]
    # line_predCatell = linReg.predict(line_age)

    regressor = svm.SVR(kernel='poly',degree=2)
    regressor.fit(Age, Cattell)
    curve_age = np.round(np.arange(Age.min()-5, Age.max() +5, .01), 2)[:, np.newaxis]
    curve_predCatell = regressor.predict(curve_age)
    plt.plot(curve_age, curve_predCatell, color="olive", linewidth=4, alpha=.7)


    plt.annotate('PearsonR= ' + str(round(rsq_cuad, 2)),
                 (20, 15), fontsize=12)
    # plt.annotate('pvalue= ' + str(round(pvalue_cuad,4)),
    #              (20, 12), fontsize=12)
    plt.annotate('pvalue < .0001',
                 (20, 12), fontsize=12)

    Residuals = returnResuduals(demographics, ['Cattell'], linReg)
    Residuals ['Intervalo'] = demographics['Intervalo']
    # dfRes_melt = pd.melt(Residuals, id_vars=['Edad'],
    #                      value_vars=['resCattell', 'Intervalo'])

    color = 'mako'

    sns.displot(data=Residuals, x='resCattell', hue='Intervalo', kind='kde',
                fill=True, palette=color)

    sns.lmplot(x='Edad', y='resCattell', data=Residuals,
               scatter=False, scatter_kws={'alpha': 0.3}, palette=color)

    sns.relplot(data=demographics, x='Cattell', y='Acer', hue='Intervalo')
    plt.title('Cattell-Acer Regression')
    rsq, pvalue = scipy.stats.pearsonr(Cattell, Acer)
    Cattell = Cattell.reshape(-1, 1)
    # Acer=Acer.reshape(-1,1)
    linReg = linear_model.LinearRegression()
    linReg.fit(Cattell, Acer)
    # Predict data of estimated models
    line_X = np.linspace(Cattell.min(), Cattell.max(), 603)[:, np.newaxis]
    line_y = linReg.predict(line_X)
    plt.plot(line_X, line_y, color="yellowgreen", linewidth=4, alpha=.7)
    plt.annotate('PearsonR= ' + str(round(rsq, 2)),
                 (20, 17), fontsize=12)
    plt.annotate('pvalue= ' + str(round(pvalue,4)),
                 (20, 10), fontsize=12)


    sns.relplot(data=demographics, x='Edad', y='Acer', hue='Intervalo')
    plt.title('Regresión de ACE-R con respecto a la Edad')

    regressor.fit(Age, Acer)
    curve_predAcer = regressor.predict(curve_age)
    plt.plot(curve_age, curve_predAcer, color="olive", linewidth=4, alpha=.7)
    plt.ylabel('ACE-R')
    Age = Age.reshape(Age.shape[0], )
    rsq, pvalue = scipy.stats.pearsonr(Age**2, Acer**2)
    plt.annotate('PearsonR= ' + str(round(rsq, 2)),
                 (20, 77), fontsize=12)
    # plt.annotate('pvalue= ' + str(round(pvalue_cuad, 4)),
    #              (20, 70), fontsize=12)
    plt.annotate('pvalue < .0001 ',
                 (20, 70), fontsize=12)

    Age = Age.reshape(Age.shape[0], )
    rsq, pvalue = scipy.stats.pearsonr(Age, Acer)
    Age = Age.reshape(-1, 1)
    # linReg = linear_model.LinearRegression()
    # linReg.fit(Age, Acer)
    # Predict data of estimated models
    # line_X = np.linspace(Age.min(), Age.max(), 603)[:, np.newaxis]
    # line_y = linReg.predict(line_X)
    # plt.plot(line_X, line_y, color="yellowgreen", linewidth=4, alpha=.7)
    # plt.annotate('PearsonR= ' + str(round(rsq, 2)),
    #              (20, 77), fontsize=12)
    # plt.annotate('pvalue= ' + str(pvalue),
    #              (20, 70), fontsize=12)
    plt.ylim([60, 110])

    plt.show()
    # return line_age, line_predCatell
    
#%% ==========================================================================
def returnResuduals(df, Variables, model):
    x = np.array(copy(df['Edad'])).reshape(-1, 1)
    resDf = copy(df)

    for var in Variables:
        nanidx = np.array(np.where(np.isnan(df[var]))).flatten()
        y = np.array(df[var].fillna(df[var].mean()))  # fill the nan with the mean value... not sure if its the best solution
        model.fit(x, y)
        # Predict data of estimated models
        predictions = model.predict(x)
        residuals = y - predictions
        resDf[var] = residuals
        resDf.loc[nanidx, var] = np.nan
        resDf.rename(columns={var: 'res' + var}, inplace=True)
    return resDf