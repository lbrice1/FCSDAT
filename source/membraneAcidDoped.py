# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 14:09:04 2020

@author: lbrice1
"""


#Support Vector Regression (SVR)
#Data from LSU blends

#Import packages
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import FormatStrFormatter
from matplotlib import cm
from data_load import data_load

#Membrane Conductivity ------------------------------------------------------#
dataCondMem = data_load('/Egmont Data - Acid doped.xlsx')

y = np.asarray(pd.DataFrame(dataCondMem, columns=['k']))
y = np.reshape(y, (len(y), ))
X = pd.DataFrame(dataCondMem, columns=['T', 'IEC'])

#Data preprocessing
scaler = StandardScaler()
X = scaler.fit_transform(X)

#SVR
#Specify model parameters
svr_rbf = SVR(kernel='rbf', C=100, gamma= 5, epsilon=0.1)

#Perform regression
sigma_model = svr_rbf.fit(X, y)
sigma_pred = sigma_model.predict(X)

def conductivityMem(T, IEC): 
    IEC = (IEC - 3.59)/1.26
    T = (T - 148.21)/61.3633
    X = np.array([T, IEC])
    X = X.reshape(1, -1)
    sigma_mem = sigma_model.predict(X)
    return sigma_mem

#End of Membrane Conductivity-------------------------------------------------#

#Ionomer Binder Conductivity and Acid Mass Fraction---------------------------#

#Load data
dataIonomer = data_load('/Fidelio Data - Acid doped.xlsx')

y0_io = np.asarray(pd.DataFrame(dataIonomer, columns=['k']))
y0_io = np.reshape(y0_io[:, 0], (len(y0_io), ))
X0_io = pd.DataFrame(dataIonomer, columns=['IEC', 'T'])
    
#Data preprocessing
scaler = StandardScaler()
X0_io = scaler.fit_transform(X0_io)

#SVR
#Specify model parameters
svr_rbf = SVR(kernel='rbf', C=100, gamma= 5, epsilon=.1)

#Perform regression
sigma_io_model = svr_rbf.fit(X0_io, y0_io)
sigma_io_pred = sigma_io_model.predict(X0_io)

#User define function to predict conductivity from blend and T
def conductivityIo(T, IEC): 
    IEC = (IEC - 8.74)/0.7714
    T = (T - 110.5)/57.65
    X0_io = np.array([T, IEC])
    X0_io = X0_io.reshape(1, -1)
    sigma_io= sigma_io_model.predict(X0_io)
    return sigma_io
#End of Binder Conductivity---------------------------------------------------#


# #User define function to visualize results------------------------------------#

def visualize(v1_min1, v1_max1, v2_min1, v2_max1, v1_min2, v1_max2, v2_min2, v2_max2, model1, model2, xlabel, ylabel):

    samples1 = np.zeros((1, 3))
    
    T = np.linspace(v1_min1, v1_max1, 200)
    DFs = np.linspace(v2_min1, v2_max1, 200)
    
    for t in T:
        for df in DFs:     
            sigma_io_i = model1(t, df)
            samples1 = np.row_stack([samples1, (t, df, sigma_io_i.item())])
    
    samples1 = np.delete(samples1, (0), axis=0)
    
    fig = plt.figure(figsize=(30, 25))
    
    axs = fig.add_subplot(2, 2, 1)
    
    axs.scatter(y, sigma_pred, s = 500, facecolors='none', edgecolors='r')
    axs.plot(y, y, color = 'k')
    axs.text(0.05, 0.95, '(a)', fontsize = 'x-large', horizontalalignment='left',verticalalignment='top', transform=axs.transAxes)
    axs.set_xlabel('$\kappa^{H^{+}}_{measured}$ (mS/cm)', fontsize = 'x-large', labelpad = 10)
    axs.set_ylabel('$\kappa^{H^{+}}_{predicted}$ (mS/cm)', fontsize = 'x-large', labelpad = 10)
    axs.set_title(' ')
    axs.tick_params(direction = 'in', pad = 20)
    
    axs = fig.add_subplot(2, 2, 3)
    axs.scatter(y0_io, sigma_io_pred, s = 500, facecolors='none', edgecolors='r')
    axs.plot(y0_io, y0_io, color = 'k')
    axs.text(0.05, 0.95, '(c)', fontsize = 'x-large', horizontalalignment='left',verticalalignment='top', transform=axs.transAxes)
    axs.set_xlabel('$\kappa^{H^{+}}_{measured}$ (mS/cm)', fontsize = 'x-large', labelpad = 10)
    axs.set_ylabel('$\kappa^{H^{+}}_{predicted}$ (mS/cm)', fontsize = 'x-large', labelpad = 10)
    axs.set_title(' ')
    axs.tick_params(direction = 'in', pad = 20)
    
    samples1 = np.zeros((1, 3))
    
    T = np.linspace(25, 220, 200)
    DFs = np.linspace(1.7, 5.4, 200)
    
    for t in T:
        for df in DFs:     
            sigma_io_i = conductivityMem(t, df)
            samples1 = np.row_stack([samples1, (t, df, sigma_io_i.item())])
    
    samples1 = np.delete(samples1, (0), axis=0)
    
    axs = fig.add_subplot(2, 2, 2)
    
    mem = plt.contourf(T, DFs, samples1[: , 2].reshape(200, 200))
    axs.text(0.05, 0.95, '(b)', fontsize = 'x-large', horizontalalignment='left',verticalalignment='top', transform=axs.transAxes)
    axs.tick_params(direction = 'in', pad = 20)
    membar = plt.colorbar(mem)
    membar.set_label('$\kappa^{H^{+}}_{predicted}$ (mS/cm)', fontsize = 'x-large', labelpad = 10)
    
    axs.set_xlabel(xlabel, fontsize = 'x-large', labelpad = 10)
    axs.set_ylabel(ylabel, fontsize = 'x-large', labelpad = 10)
    
    samples2 = np.zeros((1, 3))
    
    T = np.linspace(25, 200, 200)
    DFs = np.linspace(7.48, 9.51, 200)
    
    for t in T:
        for df in DFs:     
            sigma_io_i = conductivityIo(t, df)
            samples2 = np.row_stack([samples2, (t, df, sigma_io_i.item())])
    
    samples2 = np.delete(samples2, (0), axis=0)
    
    axs = fig.add_subplot(2, 2, 4)
    
    io = plt.contourf(T, DFs, samples2[: , 2].reshape(200, 200))
    axs.text(0.05, 0.95, '(d)', fontsize = 'x-large', horizontalalignment='left',verticalalignment='top', transform=axs.transAxes)
    axs.tick_params(direction = 'in', pad = 20)
    iobar = plt.colorbar(io)
    iobar.set_label('$\kappa^{H^{+}}_{predicted}$ (mS/cm)', fontsize = 'x-large', labelpad = 10)
    
    axs.set_xlabel(xlabel, fontsize = 'x-large', labelpad = 10)
    axs.set_ylabel(ylabel, fontsize = 'x-large', labelpad = 10)
    
    axs.tick_params(pad = 10, grid_alpha = 0.3, direction = 'in')
    axs.set_autoscale_on
    plt.rcParams['font.family']='sans-serif'
    tnfont={'fontname':'Helvetica'}
    plt.rcParams['font.size']=30
    plt.tight_layout()
    plt.savefig('MaterialsAcidDoped.pdf')
    plt.savefig('MaterialsAcidDoped.png', transparent=True)
    plt.show()
    
    return samples1, samples2
    
#End of visualization--------------------------------------------------------#

samples = visualize(25, 220, 1.7, 5.4, 25, 200, 7.48, 9.51, conductivityMem, conductivityIo, 'Temperature (°C)', 'IEC (mequiv/g)')

