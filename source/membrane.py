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
dataCondMem = data_load('/Egmont Data.xlsx')

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
dataIonomer = data_load('/Fidelio Data.xlsx')

y0_io = np.asarray(pd.DataFrame(dataIonomer, columns=['k', 'm_io']))
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
    IEC = (IEC - 1.468)/0.3481
    T = (T - 110.5)/57.65
    X0_io = np.array([T, IEC])
    X0_io = X0_io.reshape(1, -1)
    sigma_io= sigma_io_model.predict(X0_io)
    return sigma_io
#End of Binder Conductivity---------------------------------------------------#


# #User define function to visualize results------------------------------------#

def visualize(T_min1, T_max1, DF_min1, DF_max1, T_min2, T_max2, DF_min2, DF_max2, model1, model2, xlabel, ylabel):
   
    fig = plt.figure(figsize=(10, 10))
    
    axs = fig.add_subplot(2, 2, 1)
    
    #Fitness Membrane
    
    axs.scatter(y, sigma_pred, s = 200, facecolors='none', edgecolors='r')
    axs.plot(y, y, color = 'k')
    axs.text(0.05, 0.95, '(a)', fontsize = 'x-large', horizontalalignment='left',verticalalignment='top', transform=axs.transAxes)
    axs.set_xlabel('$\kappa^{H^{+}}_{measured}$ (mS/cm)', fontsize = 'x-large', labelpad = 10)
    axs.set_ylabel('$\kappa^{H^{+}}_{predicted}$ (mS/cm)', fontsize = 'x-large', labelpad = 10)
    axs.set_title(' ')
    axs.tick_params(direction = 'in', pad = 20)
    
    #Fitness ionomer
    axs = fig.add_subplot(2, 2, 3)
    axs.scatter(y0_io, sigma_io_pred, s = 200, facecolors='none', edgecolors='r')
    axs.plot(y0_io, y0_io, color = 'k')
    axs.text(0.05, 0.95, '(c)', fontsize = 'x-large', horizontalalignment='left',verticalalignment='top', transform=axs.transAxes)
    axs.set_xlabel('$\kappa^{H^{+}}_{measured}$ (mS/cm)', fontsize = 'x-large', labelpad = 10)
    axs.set_ylabel('$\kappa^{H^{+}}_{predicted}$ (mS/cm)', fontsize = 'x-large', labelpad = 10)
    axs.set_title(' ')
    axs.tick_params(direction = 'in', pad = 20)
    
    #Predictions membrane
    samples1 = np.zeros((201, ))
    samplesDF = np.zeros((1, ))
    T = np.linspace(T_min1, T_max1, 200)
    DFs = np.linspace(DF_min1, DF_max1, 200)
    
    for t in T:
        for df in DFs:     
            sigma_mem_i = conductivityMem(t, df)
            samplesDF = np.row_stack([samplesDF, (sigma_mem_i.item())])
        samples1 = np.column_stack((samples1, np.flipud(samplesDF)))
        samplesDF = np.zeros((1, ))
    
    samples1 = np.delete(samples1, (-1), axis=0)
    samples1 = np.delete(samples1, (0), axis=1)
    
    
    axs = fig.add_subplot(2, 2, 2)
    
    mem = plt.contourf(T, DFs, samples1)
    axs.text(0.05, 0.95, '(b)', fontsize = 'x-large', horizontalalignment='left',verticalalignment='top', transform=axs.transAxes)
    axs.tick_params(direction = 'in', pad = 20)
    membar = plt.colorbar(mem)
    membar.set_label('$\kappa^{H^{+}}_{predicted}$ (mS/cm)', fontsize = 'x-large', labelpad = 10)
    
    axs.set_xlabel(xlabel, fontsize = 'x-large', labelpad = 10)
    axs.set_ylabel(ylabel, fontsize = 'x-large', labelpad = 10)
    
    
    #Predictions ionomer
    samples2 = np.zeros((201, ))
    samplesDF = np.zeros((1, ))
    T = np.linspace(T_min2, T_max2, 200)
    DFs = np.linspace(DF_min2, DF_max2, 200)
    
    for t in T:
        for df in DFs:     
            sigma_io_i = conductivityIo(t, df)
            samplesDF = np.row_stack([samplesDF, (sigma_io_i.item())])
        samples2 = np.column_stack((samples2, np.flipud(samplesDF)))
        samplesDF = np.zeros((1, ))
    
    samples2 = np.delete(samples2, (-1), axis=0)
    samples2 = np.delete(samples2, (0), axis=1)
    
    axs = fig.add_subplot(2, 2, 4)
    
    io = plt.contourf(T, DFs, np.flipud(samples2))
    axs.text(0.05, 0.95, '(d)', fontsize = 'x-large', horizontalalignment='left',verticalalignment='top', transform=axs.transAxes)
    axs.tick_params(direction = 'in', pad = 20)
    iobar = plt.colorbar(io)
    iobar.set_label('$\kappa^{H^{+}}_{predicted}$ (mS/cm)', fontsize = 'x-large', labelpad = 10)
    
    axs.set_xlabel(xlabel, fontsize = 'x-large', labelpad = 10)
    axs.set_ylabel(ylabel, fontsize = 'x-large', labelpad = 10)
    
    #Additional formatting
    axs.tick_params(pad = 10, grid_alpha = 0.3, direction = 'in')
    axs.set_autoscale_on
    plt.rcParams['font.family']='sans-serif'
    tnfont={'fontname':'Helvetica'}
    plt.rcParams['font.size']=25
    plt.tight_layout()
    plt.savefig('../figures/Figure 2.pdf')
    plt.savefig('../figures/Figure 2.png', transparent=True)
    plt.show()
    
    return samples1, samples2
    
#End of visualization--------------------------------------------------------#

#samples = visualize(25, 220, 1.7, 5.4, 25, 200, 0.97, 1.96, conductivityMem, conductivityIo, 'Temperature (°C)', 'IEC (mequiv/g)')


