# -*- coding: utf-8 -*-
"""
Created on Sat Nov 14 12:35:26 2020

@author: lbrice1
"""
#Import packages
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as sps
import time

from fuelcell import Fuelcell
from fuelcell import data_load
from membrane import conductivityMem
from membrane import conductivityIo
from membrane import visualize

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 
start_time = time.time()

#Create fuel cell
#Initialize input variables
I           = np.linspace(0.01, 12, 100)
T           = 493       #K
P           = 1.59      #atm
SH2         = 1.2       #adim
SO2         = 2.2       #adim
CO_H2       = 0         #adim

A           = 5         #cm^2
IEC_mem     = 2.2        #%QPPSf
IEC_io      = 1.72
delta_mem   = 0.005     #cm
delta_io    = 0.0001    #cm
a_c         = 450       #cm^2/mgPt
L_c         = 0.5       #mgPt/cm^2

variables = np.asarray([SH2, SO2, T, P, IEC_mem, IEC_io, delta_mem, delta_io, CO_H2, L_c])
var_names = ['SH2', 'SO2', 'T', 'P', 'IEC_mem', 'IEC_io', 'delta_mem', 'delta_io', 'CO_H2', 'L_c']

#Initialize Fitted parameters
i0_an_H2_ref  = 0.144 #A/cm^2
i0_cat_ref    = 2.63e-8 #A/cm^2
alpha_an_H2   = 1.02       #dimensionless
alpha_cat     = 0.97    #dimensionless
eta           = 4.95   #dimensionless

B             = 0.09 #V
Kappa_cat     = 1.99e7   #K-A-s/cm^3-atm
Kappa_an      = Kappa_cat/2  #K-A-s/cm^3-atm

Fuelcell.f0 = 0.04

data1 = data_load('/Validation data - LBM.xlsx')
J1 = np.asarray(pd.DataFrame(data1, columns=['I (A/cm2) 220']))
J1 = J1[~np.isnan(J1).any(axis=1)]
E_exp1 = np.asarray(pd.DataFrame(data1, columns=['E (V) 220']))
E_exp1 = E_exp1[~np.isnan(E_exp1).any(axis=1)]

fuelcell1 = Fuelcell(A, IEC_mem, IEC_io, delta_mem, delta_io, L_c, a_c, E_exp1)
print('New fuel cell created')

params = np.asarray([i0_an_H2_ref, i0_cat_ref, alpha_an_H2, alpha_cat, eta, B, Kappa_cat])
params_names = ['i0_an_H2_ref', 'i0_cat_ref', 'alpha_an_H2', 'alpha_cat', 'eta', 'B', 'Kappa_cat']


data1 = data_load('/Validation data - LBM.xlsx')
J1 = np.asarray(pd.DataFrame(data1, columns=['I (A/cm2) 220']))
E_exp1 = np.asarray(pd.DataFrame(data1, columns=['E (V) 220']))
J1 = J1[~np.isnan(J1).any(axis=1)]
E_exp1 = E_exp1[~np.isnan(E_exp1).any(axis=1)]
E1 = fuelcell1.operate(J1*5, SH2, SO2, 473, P, IEC_mem, IEC_io, delta_mem, delta_io, CO_H2, L_c, params, params_names, graphs = False)[1]

data2 = data_load('/Validation data - LBM.xlsx')
J2 = np.asarray(pd.DataFrame(data2, columns=['I (A/cm2) 200']))
E_exp2 = np.asarray(pd.DataFrame(data2, columns=['E (V) 200']))
J2 = J2[~np.isnan(J2).any(axis=1)]
E_exp2 = E_exp2[~np.isnan(E_exp2).any(axis=1)]
E2 = fuelcell1.operate(J2*5, SH2, SO2, 493, P, IEC_mem, IEC_io, delta_mem, delta_io, CO_H2, L_c, params, params_names, graphs = False)[1]


data3 = data_load('/Validation data - LBM.xlsx')
J3 = np.asarray(pd.DataFrame(data3, columns=['I_CO(A/cm2)220']))
E_exp3 = np.asarray(pd.DataFrame(data3, columns=['E_CO(V)220']))
J3 = J3[~np.isnan(J3).any(axis=1)]
E_exp3 = E_exp3[~np.isnan(E_exp3).any(axis=1)]
E3 = fuelcell1.operate(J3*5, SH2, SO2, 493, P, IEC_mem, IEC_io, delta_mem, delta_io, 0.25, L_c, params, params_names, graphs = False)[1]


fig2, ax2 = plt.subplots(figsize = (13, 10))
Pex1 = E_exp1*J1
Pex2 = E_exp2*J2
Pex3 = E_exp3*J3
P1 = (np.reshape(E1, (-1, 1))*J1)
P2 = np.reshape(E2, (-1, 1))*J2
P3 = np.reshape(E3, (-1, 1))*J3

plt.scatter(Pex1[0:10, :], P1[0:10, :],label = 'Experimental (200 °C)', s = 500, facecolors='none', edgecolors='r')
plt.scatter(Pex2, P2, label = 'Experimental (220 °C)', s = 500, facecolors='none', edgecolors='b')
plt.scatter(Pex3, P3,label = 'Experimental (220 °C 25% CO)', s = 500, facecolors='none', edgecolors='g')
plt.plot(Pex1, Pex1, color = 'k')
plt.legend(loc='lower right', ncol = 1, fontsize = 'small', fancybox=False)  
plt.rcParams['font.family']='sans-serif'
tnfont={'fontname':'Helvetica'}
plt.rcParams['font.size']=35
plt.xlabel('Current density $(A/cm^2)$', fontsize = 30)
plt.ylabel('Power density $(W/cm^2)$', fontsize = 30)
plt.tick_params(direction = 'in', labelsize = 'small')
plt.tight_layout()
plt.savefig('PowerFitness.pdf') 
plt.savefig('PowerFitness.png', transparent = True)
plt.show()
