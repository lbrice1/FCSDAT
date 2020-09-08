# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 16:35:50 2020

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
alpha_an_H2   = 1       #dimensionless
alpha_cat     = 1    #dimensionless
eta           = 5   #dimensionless

B             = 0.09 #V
Kappa_cat     = 2.0e7   #K-A-s/cm^3-atm
Kappa_an      = Kappa_cat/2  #K-A-s/cm^3-atm

Fuelcell.f0 = 0.04

data1 = data_load('/Validation data - LBM.xlsx')
J1 = np.asarray(pd.DataFrame(data1, columns=['I (A/cm2) 220']))
J1 = J1[~np.isnan(J1).any(axis=1)]
E_exp1 = np.asarray(pd.DataFrame(data1, columns=['E (V) 220']))
E_exp1 = E_exp1[~np.isnan(E_exp1).any(axis=1)]

fuelcell1 = Fuelcell(A, IEC_mem, IEC_io, delta_mem, delta_io, L_c, a_c, E_exp1)
print('New fuel cell created')
#-- --------------------------------------------------------------------------|

#Show base case polariaztion curve

params = np.asarray([i0_an_H2_ref, i0_cat_ref, alpha_an_H2, alpha_cat, eta, B, Kappa_cat])
params_names = ['i0_an_H2_ref', 'i0_cat_ref', 'alpha_an_H2', 'alpha_cat', 'eta', 'B', 'Kappa_cat']

I = J1*5

polCurves = fuelcell1.operate(I, SH2, SO2, T, P, IEC_mem, IEC_io, delta_mem, delta_io, 
                              CO_H2, L_c, params, params_names, graphs = True)

#-- --------------------------------------------------------------------------|

#Global sensitivity analysis of parameters

N = 100
u = 0.01

#Perform GSA
print('Performing GSA...')

#Build the random matrices M1, M2, M3 using Monte Carlo
fuelcell1.M1 = (fuelcell1.sampling_matrix(params, N, u).transpose())
fuelcell1.M2 = (fuelcell1.sampling_matrix(params, N, u).transpose())
fuelcell1.M3 = (fuelcell1.sampling_matrix(params, N, u).transpose())

gsa = fuelcell1.gsa(I, SH2, SO2, T, P, IEC_mem, IEC_io, delta_mem, delta_io, 
                    CO_H2, L_c, params, N, params_names)

while any(t < 0 for t in gsa.flatten()):
    fuelcell1.M1 = (fuelcell1.sampling_matrix(params, N, u).transpose())
    fuelcell1.M2 = (fuelcell1.sampling_matrix(params, N, u).transpose())
    fuelcell1.M3 = (fuelcell1.sampling_matrix(params, N, u).transpose())

    gsa = fuelcell1.gsa(I, SH2, SO2, T, P, IEC_mem, IEC_io, delta_mem, 
                        delta_io, CO_H2, L_c, params, N, params_names)
    
fuelcell1.plotgsa(gsa, params_names, I, SH2, SO2, T, P, IEC_mem, IEC_io, delta_mem, 
                  delta_io, CO_H2, L_c, params)

print('GSA completed')
print("Time for GSA")
print("--- %s seconds ---" % (time.time() - start_time))
lap_time = time.time()

#-- --------------------------------------------------------------------------|


#Jaya parameter estimation using varying number of parameters
print('Performing Jaya...')

#Initialization
N = 100
max_it = 10
tol = 0.01
w = 10
z = max_it + 1 + len(params)
r = 0.3
confidence = 0.95

g1 = np.zeros((w, max_it + 4))
g2 = np.zeros((w, max_it + 6))
g3 = np.zeros((w, max_it + 8))
g4 = np.zeros((w, max_it + 8))


#Perform Jaya sequentially
lap_time = time.time()
params_base = [0.144, 2.63e-8, 1, 1, 4.8, 0.09, 2.0e7]

for j in range(w):
    
    k1 = 3
    gsa_params = fuelcell1.get_params(gsa, k1, params_base)
    params_names1 = gsa_params[0]
    params = gsa_params[1]
    
    fuelcell1.M1 = (fuelcell1.sampling_matrix(params, N, u).transpose())
    
    g1[j, :] = fuelcell1.jaya(fuelcell1.M1, fuelcell1.M3, N, I, SH2, SO2, T, P, IEC_mem, IEC_io, delta_mem, 
                              delta_io, CO_H2, L_c, params, params_names1, max_it, tol, r)

g1_mean = np.mean(g1, axis = 0)

print('3 parameters')
print("Time for Jaya - Sequential Completed")
print("--- %s seconds ---" % (time.time() - lap_time))
lap_time = time.time()
    
for j in range(w):

    k2 = 5
    gsa_params = fuelcell1.get_params(gsa, k2, params_base)
    params_names2 = gsa_params[0]
    params = gsa_params[1]
    
    fuelcell1.M1 = (fuelcell1.sampling_matrix(params, N, u).transpose())
    
    g2[j, :] = fuelcell1.jaya(fuelcell1.M1, fuelcell1.M3, N, I, SH2, SO2, T, P, IEC_mem, IEC_io, delta_mem, 
                              delta_io, CO_H2, L_c, params, params_names1, max_it, tol, r)


g2_mean = np.mean(g2, axis = 0)

print('5 parameters')
print("Time for Jaya - Sequential Completed")
print("--- %s seconds ---" % (time.time() - lap_time))
lap_time = time.time()


for j in range(w):
    
    k3 = 7
    gsa_params = fuelcell1.get_params(gsa, k3, params_base)
    params_names3 = gsa_params[0]
    params = gsa_params[1]
    
    fuelcell1.M1 = (fuelcell1.sampling_matrix(params, N, u).transpose())
    
    g3[j, :] = fuelcell1.jaya(fuelcell1.M1, fuelcell1.M3, N, I, SH2, SO2, T, P, IEC_mem, IEC_io, delta_mem, 
                              delta_io, CO_H2, L_c, params, params_names1, max_it, tol, r)


g3_mean = np.mean(g3, axis = 0)

print('7 parameters')
print("Time for Jaya - Sequential Completed")
print("--- %s seconds ---" % (time.time() - lap_time))
lap_time = time.time()

g4_mean = np.mean(g4, axis = 0)

#Obtain standard error of the mean
g1_sem = np.zeros((max_it + k1 + 1))
for j in range((max_it + k1 + 1)):
    g1_sem[j] = sps.sem(g1[:, j], axis = 0)

g2_sem = np.zeros((max_it + k2 + 1))
for j in range((max_it + k2 + 1)):
    g2_sem[j] = sps.sem(g2[:, j], axis = 0)
 
g3_sem = np.zeros((max_it + k3 + 1))
for j in range((max_it + k3 + 1)):
    g3_sem[j] = sps.sem(g3[:, j], axis = 0)
    
    
#Obtain confidence interva limits
h1 = g1_sem*sps.t.ppf((1 + confidence) / 2, w - 1)
h2 = g2_sem*sps.t.ppf((1 + confidence) / 2, w - 1)
h3 = g3_sem*sps.t.ppf((1 + confidence) / 2, w - 1)

#Plot results Confidence intervals
g1_u = g1_mean + h1
g2_u = g2_mean + h2
g3_u = g3_mean + h3

g1_l = g1_mean - h1
g2_l = g2_mean - h2
g3_l = g3_mean - h3

obj_func_values1 = g1_mean[k1:-1]
obj_func_values1_u = g1_u[k1:-1]
obj_func_values1_l = g1_l[k1:-1]

obj_func_values2 = g2_mean[k2:-1]
obj_func_values2_u = g2_u[k2:-1]
obj_func_values2_l = g2_l[k2:-1]

obj_func_values3 = g3_mean[k3:-1]
obj_func_values3_u = g3_u[k3:-1]
obj_func_values3_l = g3_l[k3:-1]

x_coord = np.arange(1, max_it + 1)

plt.figure(figsize = (15, 10))

plt.plot(x_coord, obj_func_values1, label = '3 parameters', color = 'purple')
plt.fill_between(x_coord, obj_func_values1_u, obj_func_values3_l, color = 'purple', alpha = 0.2)

plt.plot(x_coord, obj_func_values2, label = '5 parameters', color = 'teal')
plt.fill_between(x_coord, obj_func_values2_u, obj_func_values2_l, color = 'teal', alpha = 0.2)

plt.plot(x_coord, obj_func_values3, label = '7 parameters', color = 'limegreen')
plt.fill_between(x_coord, obj_func_values3_u, obj_func_values3_l, color = 'limegreen', alpha = 0.2)

plt.legend(fontsize = 'small', loc='upper right')
plt.rcParams['font.family']='sans-serif'
tnfont={'fontname':'Helvetica'}
plt.rcParams['font.size']=20
plt.xlabel('Number of iterations')
plt.ylabel('Standard Error')
plt.xlim(1, max_it)
plt.tick_params(direction = 'in')
plt.tight_layout()

plt.savefig('Jaya.pdf')  
plt.savefig('Jaya.png')
plt.show()

print('Jaya completed')
    
#- --------------------------------------------------------------------------|

params = g3_mean[0:k3]

polCurves = fuelcell1.showFit(I, SH2, SO2, T, P, IEC_mem, IEC_io, delta_mem, delta_io, CO_H2, L_c, params, params_names3, graphs = False, overpotential = 'Contributions')

Global sensitivity analysis of variables

N = 100
u = 0.4

#Perform GSA
print('Performing GSA...')

#Build the random matrices M1, M2, M3 using Monte Carlo
fuelcell1.M1 = (fuelcell1.sampling_matrix(variables, N, u).transpose())
fuelcell1.M2 = (fuelcell1.sampling_matrix(variables, N, u).transpose())
fuelcell1.M3 = (fuelcell1.sampling_matrix(variables, N, u).transpose())

gsaV = fuelcell1.gsaV(I, variables, params, N, params_names3)

while any(t < 0 for t in gsaV.flatten()):
    fuelcell1.M1 = (fuelcell1.sampling_matrix(variables, N, u).transpose())
    fuelcell1.M2 = (fuelcell1.sampling_matrix(variables, N, u).transpose())
    fuelcell1.M3 = (fuelcell1.sampling_matrix(variables, N, u).transpose())
    
    gsaV = fuelcell1.gsaV(I, variables, params, N, params_names3)
    
    print('I tried')
    
fuelcell1.plotgsaV(gsaV, var_names, params_names3, I, SH2, SO2, T, P, 
                  IEC_mem, IEC_io, delta_mem, delta_io, CO_H2, L_c, params)

print('GSAV completed')
print("Time for GSAV")
print("--- %s seconds ---" % (time.time() - start_time))
lap_time = time.time()

#- --------------------------------------------------------------------------|

#Plot exploration with optimized parameters

polCurves = fuelcell1.exploreConfigs(I, SH2, SO2, T, P, IEC_mem, IEC_io, delta_mem, delta_io, CO_H2, L_c, params, params_names3, graphs = False, overpotential = 'Contributions')

print("Total time elapsed")
print("--- %s seconds ---" % (time.time() - start_time))
- --------------------------------------------------------------------------|

Generate dataset for clustering---------------------------------------------|

p_size = 6000
s_size = 5000
interval = [0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4]
polcurves = fuelcell1.data_gen(p_size, 
                               s_size, params, 
                               params_names3, 
                               interval)

#--------------------------------End of main--------------------------------##