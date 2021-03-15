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
from createFuelCell import createfc
from membrane import conductivityMem
from membrane import conductivityIo
from membrane import visualize

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 
start_time = time.time()

#Create fuel cell
I               = np.linspace(0.01, 12, 100)
fuelcell1, J1, i0_an_H2_ref, i0_cat_ref, alpha_an_H2, alpha_cat, eta, B, Kappa_cat, variables, var_names  = createfc(I)

#-- --------------------------------------------------------------------------|
#Initialize input variables
#Operating conditions
T           = 493       #K
P           = 1.59      #atm
SH2         = 1.2       #adim
SO2         = 2.2       #adim
CO_H2       = 0         #adim

#Cell design
A           = 5         #cm^2
IEC_mem     = 2.2        #%QPPSf
IEC_io      = 1.72
delta_mem   = 0.005     #cm
delta_io    = 0.0001    #cm
a_c         = 450       #cm^2/mgPt
L_c         = 0.5       #mgPt/cm^2

params       = np.asarray([i0_an_H2_ref, i0_cat_ref, alpha_an_H2, alpha_cat, eta, B, Kappa_cat])
params_names = ['i0_an_H2_ref', 'i0_cat_ref', 'alpha_an_H2', 'alpha_cat', 'eta', 'B', 'Kappa_cat']

I = J1*5

polCurves   = fuelcell1.operate(I, SH2, SO2, T, P, IEC_mem, IEC_io, delta_mem, delta_io, 
                              CO_H2, L_c, params, params_names, graphs = True)

#-- --------------------------------------------------------------------------|
#Examples
#Global sensitivity analysis of parameters
gsa = fuelcell1.performGSA(I, SH2, SO2, T, P, IEC_mem, IEC_io, delta_mem, delta_io, CO_H2, L_c, params, params_names)

#Jaya parameter estimation using varying number of parameters
params, params_names3 = fuelcell1.performJaya(params, gsa, I, SH2, SO2, T, P, IEC_mem, IEC_io, delta_mem, delta_io, CO_H2, L_c)

#Plot individual activation overpotentials
fuelcell1.plotActivation(I, SH2, SO2, T, P, IEC_mem, IEC_io, delta_mem, delta_io, CO_H2, L_c, params, params_names3)

#Global sensitivity analysis of variables
fuelcell1.performGSA_V(I, variables, var_names, params, params_names3)

#Plot exploration with optimized parameters
fuelcell1.performExplore(I, SH2, SO2, T, P, IEC_mem, IEC_io, delta_mem, delta_io, CO_H2, L_c, params, params_names3)

