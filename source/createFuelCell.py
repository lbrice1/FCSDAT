
import numpy as np
import pandas as pd
from data_load import data_load
from fuelcell import Fuelcell

def createfc(I):
    """
    

    Parameters
    ----------
    I : TYPE
        DESCRIPTION.

    Returns
    -------
    cell1 : TYPE
        DESCRIPTION.

    """
    
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
    L_c      = 0.5       #mgPt/cm^2

    
    variables = np.asarray([SH2, SO2, T, P, IEC_mem, IEC_io, delta_mem, delta_io, CO_H2, L_c])
    var_names = ['SH2', 'SO2', 'T', 'P', 'IEC_mem', 'IEC_io', 'delta_mem', 'delta_io', 'CO_H2', 'L_c']
    
    #Base case data
    data1 = data_load('/Validation data - LBM.xlsx')
    J1 = np.asarray(pd.DataFrame(data1, columns=['I (A/cm2) 220']))
    J1 = J1[~np.isnan(J1).any(axis=1)]
    E_exp1 = np.asarray(pd.DataFrame(data1, columns=['E (V) 220']))
    E_exp1 = E_exp1[~np.isnan(E_exp1).any(axis=1)]
    
    #Create fuel cell
    cell1 = Fuelcell(A, IEC_mem, IEC_io, delta_mem, delta_io, L_c, a_c, E_exp1)
    
    params = np.asarray([cell1.i0_an_H2_ref, cell1.i0_cat_ref, cell1.alpha_an_H2, cell1.alpha_cat, cell1.eta, cell1.B, cell1.Kappa_cat])
    params_names = ['i0_an_H2_ref', 'i0_cat_ref', 'alpha_an_H2', 'alpha_cat', 'eta', 'B', 'Kappa_cat']
    polCurves = cell1.operate(J1*5, SH2, SO2, T, P, IEC_mem, IEC_io, 
                              delta_mem, delta_io, CO_H2, L_c, params, params_names, graphs = True)
    
    print('New fuel cell created')
    
    return cell1, J1, cell1.i0_an_H2_ref, cell1.i0_cat_ref, cell1.alpha_an_H2, cell1.alpha_cat, cell1.eta, cell1.B, cell1.Kappa_cat, variables, var_names