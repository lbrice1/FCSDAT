# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 16:11:22 2020

@author: lbrice1
"""

import numpy as np
from numpy import sqrt, exp, log
import matplotlib.pyplot as plt
import pandas as pd
import random
from datetime import date


from membrane import conductivityMem
from membrane import conductivityIo
from data_load import data_load

class Fuelcell:
    
    #Create the fuel cell instance
    def __init__(self, A, IEC_mem, IEC_io, delta_mem, delta_io, L_c, a_c, E_exp1):
        self.A = A
        self.IEC_mem = IEC_mem
        self.IEC_io = IEC_io
        self.delta_mem = delta_mem
        self.delta_io = delta_io
        self.L_c = L_c
        self.a_c = a_c
        self.E_exp1 = E_exp1[~np.isnan(E_exp1).any(axis=1)]
        
        self.f0 = 0
        
        self.M1 = None
        self.M2 = None
        self.M3 = None
        
        self.Nq = None
        self.NTq = None
        
    #Operate de fuel cell at given conditions

    def operate(self, I, SH2, SO2, T, P, IEC_mem, IEC_io, delta_mem, delta_io, CO_H2, L_c, params, params_names, graphs = False, gsa = [0, 1, 2, 3, 4, 5, 6]):
        
        params_fixed = ([0.144, 2.63e-8, 1, 1, 4.8, 0.09, 2.0e7])
        
        params_names_orig = ['i0_an_H2_ref', 'i0_cat_ref', 'alpha_an_H2', 'alpha_cat', 'eta', 'B', 'Kappa_cat']
        paramsDict = {params_names_orig[0]: 0, 
                      params_names_orig[1]: 1, 
                      params_names_orig[2]: 2, 
                      params_names_orig[3]: 3, 
                      params_names_orig[4]: 4, 
                      params_names_orig[5]: 5, 
                      params_names_orig[6]: 6}

        for j in range(len(params_names)):
            params_fixed[int(paramsDict[params_names[j]])] = params[j]
        
        i0_an_H2_ref  = params_fixed[0]
        i0_cat_ref    = params_fixed[1]
        alpha_an_H2   = params_fixed[2]
        alpha_cat     = params_fixed[3]
        eta           = params_fixed[4]
        B             = params_fixed[5]
        Kappa_cat     = params_fixed[6]
        
        a_D           = 4500
        b_D           = 10000
        c_D           = 4010
        a_H           = 900
        b_H           = 1500
        c_H           = 650
        
        Kappa_an      = Kappa_cat/2
            
        A           = self.A         #cm^2
        a_c         = self.a_c       #cm^2/mgPt
          
        Tcat    = T         #K
        Tan     = T         #K
        
        Pan     = P         #atm
        Pcat    = P         #atm
        
        #Model constants
        F           = 96485 #J/mol
        R1          = 8.314 #J/molK
    
        #Model parameters
        E_c_O2      = 72400 #J/mol
        E_c_H2      = 16900 #J/mol
    
        zH2           = 2
        zO2           = 4
        zH2O          = 2
       
        E = []
        E_up = []
        E_low = []
        Eoc = []
        Eact = []
        Econ = []
        Eohm = []
        Power = []
        
        k = 0
        
        #Build polarization curve
        for k in range(len(I)):
            i = I[k]/A
                       
            #Mole balances
            #Steady state        
            NH2     = (SH2 - 1)*I[k]/zH2/F          #mol/s
            NCO     = (SH2 - 1)*I[k]/zH2/F*CO_H2    #mol/s
            NO2     = (SO2 - 1)*I[k]/zO2/F          #mol/s
            NH2Ocat = I[k]/zH2O/F                   #mol/s
            NCO2an  = I[k]/zH2/F*CO_H2              #mol/s
                            
            nH2an           = NH2/A         #mol/s-cm^2
            nCOan           = NCO/A         #mol/s-cm^2
            nO2cat          = NO2/A         #mol/s-cm^2
            nH2Ocat         = NH2Ocat/A     #mol/s-cm^2
            nCO2an          = NCO2an/A
            
            xH2an           = nH2an/(nH2an + nCOan + nCO2an)     #molar fraction
            xH2an           = 1/(1 + CO_H2)
            xCOan           = nCOan/(nH2an + nCOan + nCO2an)     #molar fraction
            xO2cat          = nO2cat/(nO2cat + nH2Ocat) #molar fraction
            xH2Ocat         = nH2Ocat/(nO2cat + nH2Ocat)#molar fraction
            
            PH2             = Pan*xH2an     #atm
            PO2             = Pcat*xO2cat   #atm
            PH2O            = Pcat*xH2Ocat  #atm
            
            #Compute overpotentials
            Eoc_i           = (1.23 - 0.9e-3*(T - 298.15) + 2.3*R1*T/4/F*(2*log(PH2) + log(PO2) - 2*log(PH2O))).item() #V #Rasheed et al.
            
            CH2_0_ref = 2.11e-7 #mol/cm^3
            CO2_0_ref = 1.07e-7 #mol/cm^3
                    
            m_io      = 0.0902*IEC_io + 0.0352
            
            D_O2_PBI     = (1.0e-6)*exp(-(a_D*m_io**2-b_D*m_io+c_D)/T) 
            D_H2_PBI     = 2*D_O2_PBI
                                                                                                                            
            
            H_O2_PBI     = (1.0e-6)*exp(-(a_H*m_io**2-b_H*m_io+c_H)/T)
            H_H2_PBI     = 4*H_O2_PBI
            
            i0_an_H2     = ((PH2*H_H2_PBI/CH2_0_ref)**0.5)*a_c*L_c*exp(-E_c_H2/R1/Tan*(1 - Tan/433))*i0_an_H2_ref*exp(-eta*(1 - m_io)) #A/cm^2 #Corrected using Barbir's book
            i0_cat       = ((PO2*H_O2_PBI/CO2_0_ref)**1)*a_c*L_c*exp(-E_c_O2/R1/Tcat*(1 - Tcat/423))*i0_cat_ref*exp(-eta*(1 - m_io))   #A/cm^2 #Correction for phosphoric acid, from Cheddie et al.
                                                                                                                        
            if CO_H2 > 0:
                thetaCO  = 19.9*exp((-7.69e-3)*Tan) + 0.085*log(CO_H2) 
            else:
                thetaCO  = 0
            
            i0_an_CO     = i0_an_H2*(1 - thetaCO)**2
            
            Eact_an      = R1*T/alpha_an_H2/F*log(i/i0_an_CO)   #V
            Eact_cat     = R1*T/alpha_cat/F*log(i/i0_cat)       #V
            Eact_i       = Eact_an + Eact_cat                   #V
                    
            i_max_cat     = Kappa_cat*D_O2_PBI*PO2/T/delta_io                 #A/cm^2
            i_max_an      = Kappa_an*D_H2_PBI*PH2/T/delta_io*(1 - thetaCO)**2 #A/cm^2
            
            Econ_an       = -B*log(1 - i/i_max_an)   #V
            Econ_cat      = -B*log(1 - i/i_max_cat)  #V
            
            Econ_i        = Econ_an + Econ_cat      #V
            
            sigma_mem    = (conductivityMem((T - 273), IEC_mem)).item()/1000             #S/cm
            sigma_io     = (conductivityIo((T - 273), IEC_io)).item()/1000              #S/cm
            Eohm_i       = delta_mem*I[k]/A/sigma_mem + 2*delta_io*I[k]/A/sigma_io      #V
            
            E_i          = Eoc_i - Eact_i - Econ_i - Eohm_i   #V
            E_i_up       = E_i + self.f0
            E_i_low      = E_i - self.f0
            Power_i      = I[k]/A*E_i
        
            E.append(np.asscalar(E_i))
            E_up.append(np.asscalar(E_i_up))
            E_low.append(np.asscalar(E_i_low))
            Eoc.append(Eoc_i)
            Eact.append(np.asscalar(Eact_i))
            Econ.append(np.asscalar(Econ_i))
            Eohm.append(np.asscalar(Eohm_i))
            Power.append(Power_i.astype(float))
            
        if graphs == True:   
            
            data1 = data_load('/Validation data - LBM.xlsx')
            J1 = np.asarray(pd.DataFrame(data1, columns=['I (A/cm2) 220']))
            J1 = J1[~np.isnan(J1).any(axis=1)]
            E_exp1 = np.asarray(pd.DataFrame(data1, columns=['E (V) 220']))
            E_exp1 = E_exp1[~np.isnan(E_exp1).any(axis=1)]
            
            fig, ax1 = plt.subplots(figsize = (15, 11))
            ax1.plot(I/A, E, label = 'Model', linewidth = 3, color = 'teal')
            ax1.fill_between(np.reshape(I/A, (17,)), E_up, E_low, color = 'teal', alpha = 0.2)
            ax1.scatter(J1, E_exp1, label = 'Experimental (220 °C, 1.6 atm)', marker = 'D', color = 'limegreen')
            
            plt.ylim(0, 1.0)
            plt.xlim(0, 2.5)
            plt.legend(fontsize = 'xx-small', loc='upper right', ncol = 3)  
            plt.rcParams['font.family']='sans-serif'
            tnfont={'fontname':'Helvetica'}
            plt.rcParams['font.size']=35
            plt.xlabel('Current intensity $(A/cm^2)$')
            plt.ylabel('Voltage $(V)$')
            plt.tight_layout()
            plt.tick_params(direction = 'in')
            plt.savefig('Operation.pdf')
            plt.savefig('Operation.png')
            plt.show()
        
        return Eoc, E, Eact, Econ, Eohm, E_up, E_low
    
    def showFit(self, I, SH2, SO2, T, P, IEC_mem, IEC_io, delta_mem, delta_io, CO_H2, L_c, params, params_names, graphs = False, overpotential = 'E'):
        I = np.linspace(0.01, 12, 100)

        polCurves = []
        
        T = 473
        polCurve_q = np.asarray(self.operate(I, SH2, SO2, T, P, IEC_mem, IEC_io, delta_mem, delta_io, CO_H2, L_c, params, params_names, graphs = False))
        polCurves.append(polCurve_q)
        
        T = 493
        polCurve_q = np.asarray(self.operate(I, SH2, SO2, T, P, IEC_mem, IEC_io, delta_mem, delta_io, CO_H2, L_c, params, params_names, graphs = False))
        polCurves.append(polCurve_q)
        
        CO_H2 = 0.25
        polCurve_q = np.asarray(self.operate(I, SH2, SO2, T, P, IEC_mem, IEC_io, delta_mem, delta_io, CO_H2, L_c, params, params_names, graphs = False))
        polCurves.append(polCurve_q)
        
        polCurves = np.asarray(polCurves)
        
        data1 = data_load('/Validation data - LBM.xlsx')
        J1 = np.asarray(pd.DataFrame(data1, columns=['I (A/cm2) 220']))
        E_exp1 = np.asarray(pd.DataFrame(data1, columns=['E (V) 220']))
    
        data2 = data_load('/Validation data - LBM.xlsx')
        J2 = np.asarray(pd.DataFrame(data2, columns=['I (A/cm2) 200']))
        E_exp2 = np.asarray(pd.DataFrame(data2, columns=['E (V) 200']))
        
        data3 = data_load('/Validation data - LBM.xlsx')
        J3 = np.asarray(pd.DataFrame(data3, columns=['I_CO(A/cm2)220']))
        E_exp3 = np.asarray(pd.DataFrame(data3, columns=['E_CO(V)220']))
        
        tags = ['(200 °C, 0% CO)', '(220 °C, 0% CO)', '(220 °C, 25% CO)']
        
        labels = []
        linestyle = ['-', '-', '-.', ':', '-']
        temp = [tag for tag in tags]  
        
        overpotentialDict = {'Eoc': 0, 'E': 1, 'Eact': 2, 'Econ': 3, 'Eohm': 4, 'Contributions': 5, 'Power' : 6}
        overpotential = overpotentialDict[overpotential]
        
        overpotentialsLabelsDict = {'$E_{oc}$': 0, 'E': 1, '$E_{act}$': 2, '$E_{con}$': 3, '$E_{ohm}$': 4, 'Contributions': 5, 'Power' : 6}
        reverseoverpotentialsLabelsDict = {w: r for r, w in overpotentialsLabelsDict.items()}
        
        plt.figure(figsize = (15, 11))
        
        colormap = ['teal', 'limegreen', 'purple' ]
        
        for ip in range(1, overpotential):
            for p in range(len(tags)):
                plt.plot(I/self.A, polCurves[p, ip, :], linestyle = linestyle[ip], linewidth = 3, color = colormap[p])
        for p in range(len(tags)):
            plt.fill_between(np.reshape(I/self.A, (len(I),)), polCurves[p, 5, :], polCurves[p, 6, :], color = colormap[p], alpha = 0.2)
            labels.append('Model ' + temp[p])
            
        angle = 30
        
        plt.text(0.25, 0.3, 'Activation overpotentials', fontsize = 'medium', rotation=20)
        plt.text(1.0, 0.11, 'Concentration overpotentials', fontsize = 'medium', rotation=20)
        plt.text(1.5, 0.05, 'Ohmic overpotentials', fontsize = 'medium', rotation=5)

        
        labels = [item for item in labels]
        
        labels.append('Overpotential (200 °C 0% CO)')          
        plt.scatter(J2, E_exp2, label = 'Experimental (200 °C)', marker = '^', color = colormap[0])
        
        labels.append('Overpotential (220 °C 0% CO)')          
        plt.scatter(J1, E_exp1, label = 'Experimental (220 °C)', marker = 'x', color = colormap[1])
        
        labels.append('Overpotential (220 °C 25% CO)')          
        plt.scatter(J3, E_exp3, label = 'Experimental (200 °C)', marker = '^', color = colormap[2])
        
        
        
        plt.ylim(0, 1.0)
        plt.xlim(0, 2.5)
        plt.legend(labels, loc='upper right', ncol = 2, fontsize = 'medium', fancybox=True)  
        plt.rcParams['font.family']='sans-serif'
        tnfont={'fontname':'Helvetica'}
        plt.rcParams['font.size']=35
        plt.xlabel('Current intensity $(A/cm^2)$', fontsize = 30)
        plt.ylabel('Voltage $(V)$', fontsize = 30)
        plt.tick_params(direction = 'in', labelsize = 'small')
        plt.tight_layout()
        plt.savefig('Fitness.pdf') 
        plt.savefig('Fitness.png', transparent = True)
        plt.show()
        return polCurves
    
    def exploreConfigs(self, I, SH2, SO2, T, P, IEC_mem, IEC_io, delta_mem, delta_io, CO_H2, L_c, params, params_names, graphs = False, overpotential = 'E'):
        
        I = np.linspace(0.01, 12, 100)
        
        polCurves = []
        
        #Base case
        T           = 493       #K
        P           = 1.59      #atm
        CO_H2       = 0         #adim
        delta_mem   = 0.005     #cm
        L_c         = 0.5       #mgPt/cm^2
        
        polCurve_0 = np.asarray(self.operate(I, SH2, SO2, T, P, IEC_mem, IEC_io, delta_mem, delta_io, CO_H2, L_c, params, params_names, graphs = False))
        polCurves.append(polCurve_0)
        
        #Increase temperature
        polCurve_1 = np.asarray(self.operate(I, SH2, SO2, 513, P, IEC_mem, IEC_io, delta_mem, delta_io, CO_H2, L_c, params, params_names, graphs = False))
        polCurves.append(polCurve_1)
        
        #Decrease Pt loading
        polCurve_2 = np.asarray(self.operate(I, SH2, SO2, T, P, IEC_mem, IEC_io, delta_mem, delta_io, CO_H2, 0.3, params, params_names, graphs = False))
        polCurves.append(polCurve_2)
        
        #Increase P
        polCurve_3 = np.asarray(self.operate(I, SH2, SO2, T, 2, IEC_mem, IEC_io, delta_mem, delta_io, CO_H2, L_c, params, params_names, graphs = False))
        polCurves.append(polCurve_3)

        #Decrease delta_mem
        polCurve_4 = np.asarray(self.operate(I, SH2, SO2, T, P, IEC_mem, IEC_io, 0.003, delta_io, CO_H2, L_c, params, params_names, graphs = False))
        polCurves.append(polCurve_4)
    
        #10 % CO
        polCurve_5 = np.asarray(self.operate(I, SH2, SO2, T, P, IEC_mem, IEC_io, delta_mem, delta_io, 0.1, L_c, params, params_names, graphs = False))
        polCurves.append(polCurve_5)
        
        polCurves = np.asarray(polCurves)
        
        
        tags = [' $Base$ $case$', ' $T=240$% $°C$', ' $L_c=0.3$ $mg/cm^{2}$', ' $P=2$ $atm$', ' $\u03B4_{mem}=0.003$ $mm$', ' $CO/H_2$=0.1']
        
        labels = []
        linestyle = ['-', '-', '-.', ':', '-']
        temp = [tag for tag in tags]  
        
        overpotentialDict = {'Eoc': 0, 'E': 1, 'Eact': 2, 'Econ': 3, 'Eohm': 4, 'Contributions': 5, 'Power' : 6}
        overpotential = overpotentialDict[overpotential]
        overpotentialsLabelsDict = {'$E_{oc}$': 0, '$E$': 1, '$E_{act}$': 2, '$E_{con}$': 3, '$E_{ohm}$': 4, 'Contributions': 5, 'Power' : 6}
        reverseoverpotentialsLabelsDict = {w: r for r, w in overpotentialsLabelsDict.items()}
        
        fig, ax1 = plt.subplots(figsize = (20, 15))
        
        colormap = ['r', 'b', 'g', 'c', 'y', 'm']
        
        for ip in range(2, overpotential):
            for p in range(len(tags)):
                ax1.plot(I/self.A, polCurves[p, ip, :], linestyle = linestyle[ip], linewidth = 3, color = colormap[p])
                labels.append(reverseoverpotentialsLabelsDict[ip] + temp[p])
        
        ax2 = ax1.twinx()
        
        for p in range(len(tags)):
            ax2.plot(I/self.A, polCurves[p, 1, :]*I/self.A, linewidth = 3, color = colormap[p])
            labels.append(temp[p])
            
        
        labels = [item for item in labels]
        
        
        ax1.set_ylim(0, 1.0)
        ax1.set_xlim(0, 2.5)
        ax1.legend(labels, fontsize = 'xx-small', loc='upper right', ncol = 3)  
        plt.rcParams['font.family']='sans-serif'
        tnfont={'fontname':'Helvetica'}
        plt.rcParams['font.size']=35
        ax1.set_xlabel('Current intensity $(A/cm^2)$')
        ax1.set_ylabel('Voltage $(V)$')
        ax2.set_ylabel('Power density $(W/cm^2)$')
        plt.tick_params(direction = 'in')
        plt.tight_layout()
        plt.savefig('Exploration.pdf') 
        plt.show()
        
        return polCurves
    
    
    def sampling_matrix(self, params, N, u):
        M_arr = np.zeros((len(params), N))
        for j in range(0, len(params)):
            param_j = params[j]
            lower_bound  = param_j - param_j*u
            upper_bound = param_j + param_j*u
            row = np.zeros((N,))
            for i in range(0, N):
                z = random.uniform(lower_bound, upper_bound)
                row[i] = z
            M_arr[j, :] = row
        M = np.reshape(M_arr, (len(params), N))
        
        return M

    #Generate Nq matrices
    def Nmatrix(self, params, N):
        Nq_list  = np.zeros((N, len(params), len(params)))
        NTq_list = np.zeros((N, len(params), len(params)))
        
        for q in range(len(params)):
            self.Nq = self.M2.copy()
            self.Nq[:, q] = self.M1.copy()[:, q]
            Nq_list[:, q, :] = self.Nq
            self.NTq = self.M1.copy()
            self.NTq[:, q] = self.M2.copy()[:, q]
            NTq_list[:, q, :] = self.NTq
 
        return Nq_list, NTq_list
    
    
    #Evaluate the row vectors of sampling matrix
    def evaluation(self, M, M3, N, I, SH2, SO2, T, P, IEC_mem, IEC_io, delta_mem, delta_io, CO_H2, L_c, params_names, jaya = False):
        #Takes an sample matrix and maps it into a vector of objective function outputs
        g = np.zeros((N, 1))
        k = 0
        n = 0
        while n < len(M[:, 0]):
            g_i = sqrt(sum(((np.asarray(self.operate(I, SH2, SO2, T, P, IEC_mem, IEC_io, delta_mem, delta_io, CO_H2, L_c, M[n, :], params_names))[1, :] - np.reshape(self.E_exp1, (len(self.E_exp1, ))))**2))/(len(I) - 1)/len(I))
            
            if jaya == False:
                if np.isnan(g_i):
                    while np.isnan(g_i):
                        g_i = sqrt(sum(((np.asarray(self.operate(I, SH2, SO2, T, P, IEC_mem, IEC_io, delta_mem, delta_io, CO_H2, L_c, M3[k, :], params_names))[1, :] - np.reshape(self.E_exp1, (len(self.E_exp1, ))))**2))/(len(I) - 1)/len(I))
                        k += 1
                    self.M1[n, :] = self.M3[k, :]
                    
            g[n] = g_i
            n += 1
        g = np.asarray(g)
    
        return g
    
    #Evaluate the row vectors of sampling matrix
    def evaluationV(self, M, M3, N, I, params, params_names, jaya = False):
        #Takes an sample matrix and maps it into a vector of objective function outputs
        g = np.zeros((N, 1))
        k = 0
        n = 0
        while n < len(M[:, 0]):
            g_i = max(np.asarray(self.operate(I, M[n, 0], M[n, 1], M[n, 2], M[n, 3], M[n, 4], M[n, 5], M[n, 6], M[n, 7], M[n, 8], M[n, 9], params, params_names))[1, :]*np.reshape(I/self.A, (len(I), )))
            
            if jaya == False:
                if np.isnan(g_i):
                    while np.isnan(g_i):
                        g_i = max(np.asarray(self.operate(I, M3[n, 0], M3[n, 1], M3[n, 2], M3[n, 3], M3[n, 4], M3[n, 5], 
                                                          M3[n, 6], M3[n, 7], M3[n, 8], M3[n, 9], params, params_names)[1, :]*np.reshape(I/self.A, (len(I), ))))
                        k += 1
                    self.M1[n, :] = self.M3[k, :]
                    
            g[n] = g_i
            n += 1
        g = np.asarray(g)
    
        return g
    
    def evaluationN(self, M, N, I, SH2, SO2, T, P, IEC_mem, IEC_io, delta_mem, delta_io, CO_H2, L_c, params_names):
        g = np.zeros((N, 1))
        for n in range(N):
            g_i = sqrt(sum(((np.asarray(self.operate(I, SH2, SO2, T, P, IEC_mem, IEC_io, delta_mem, delta_io, CO_H2, L_c, M[n, :], params_names))[1, :] - np.reshape(self.E_exp1, (len(self.E_exp1, ))))**2))/(len(I) - 1)/len(I))
            
            if np.isnan(g_i):
                g_i = self.f0
            g[n] = g_i
        g = np.asarray(g)
    
        return g
    
    def evaluationNV(self, M, N, I, params, params_names):
        g = np.zeros((N, 1))
        for n in range(N):
            g_i = max(np.asarray(self.operate(I, M[n, 0], M[n, 1], M[n, 2], M[n, 3], M[n, 4], M[n, 5], M[n, 6], M[n, 7], M[n, 8], M[n, 9], params, params_names))[1, :]*np.reshape(I/self.A, (len(I), )))
            
            if np.isnan(g_i):
                g_i = self.f0
            g[n] = g_i
        g = np.asarray(g)
    
        return g
    
    #Obtain indices
    def sortSecond(self, val): 
        return val[1]  
    
    def sortLast(self, val): 
        return val[-1]  
    
    #Obtain output vectors
    def gsa(self, I, SH2, SO2, T, P, IEC_mem, IEC_io, delta_mem, delta_io, CO_H2, L_c, params, N, params_names):

        Sj_list_arr = np.zeros((len(params), 2))
        STj_list_arr = np.zeros((len(params), 2))
        
        g =  np.reshape(self.evaluation(self.M1, self.M3, N, I, SH2, SO2, T, P, IEC_mem, IEC_io, delta_mem, delta_io, CO_H2, L_c, params_names), (N,))
        gR =  np.reshape(self.evaluation(self.M2, self.M3, N, I, SH2, SO2, T, P, IEC_mem, IEC_io, delta_mem, delta_io, CO_H2, L_c, params_names), (N,))
        self.f0 = (np.mean(g) + np.mean(gR))/2
        
        for q in range(len(params)):

            Nq_NTq = self.Nmatrix(params, N)
            Nq = Nq_NTq[0]
            NTq = Nq_NTq[1]
            
            gq =  np.reshape(self.evaluationN(Nq[:, q, :], N, I, SH2, SO2, T, P, IEC_mem, IEC_io, delta_mem, delta_io, CO_H2, L_c, params_names), (N,))
            gRq =  np.reshape(self.evaluationN(NTq[:, q, :], N, I, SH2, SO2, T, P, IEC_mem, IEC_io, delta_mem, delta_io, CO_H2, L_c, params_names), (N,))
            
            #Obtain estimators
            V_sum = 0
            for n in range(N):
                V_sum = (g[n]**2 + gR[n]**2) + V_sum
            V = V_sum/2/N - self.f0**2
            
            gamma_sq_q_sum = 0
            for n in range(N):
                gamma_sq_q_sum = ((g[n]*gR[n]) + (gq[n]*gRq[n])) + gamma_sq_q_sum
            gamma_sq_q = gamma_sq_q_sum/2/N
            
            Vj = (np.dot(g, gq) + np.dot(gRq, gR))/2/N - gamma_sq_q
            
            Vmj = (np.dot(gRq, g) + np.dot(gq, gR))/2/N - gamma_sq_q
            
            #Obtain indices
            Sj = Vj/V
            STj = 1 - Vmj/V
            
            Sj_list_arr[q, :] = ([q, Sj])
            STj_list_arr[q, :] = ([q, STj])
    
        Sj_list_arr = sorted(Sj_list_arr, key = self.sortSecond, reverse = True)
        STj_list_arr = sorted(STj_list_arr, key = self.sortSecond, reverse = True)
        
        Sj_list_arr = np.array(Sj_list_arr)
        STj_list_arr = np.array(STj_list_arr)
        
        gsa = np.column_stack((Sj_list_arr, STj_list_arr))
        
        return gsa
    
    #Obtain output vectors (variable analysis)
    def gsaV(self, I, variables, params, N, params_names):

        Sj_list_arr = np.zeros((len(variables), 2))
        STj_list_arr = np.zeros((len(variables), 2))
        
        g =  np.reshape(self.evaluationV(self.M1, self.M3, N, I, params, params_names), (N,))
        gR =  np.reshape(self.evaluationV(self.M2, self.M3, N, I, params, params_names), (N,))
        f0 = (np.mean(g) + np.mean(gR))/2
        
        for q in range(len(variables)):

            Nq_NTq = self.Nmatrix(variables, N)
            Nq = Nq_NTq[0]
            NTq = Nq_NTq[1]
            
            gq =  np.reshape(self.evaluationNV(Nq[:, q, :], N, I, params, params_names), (N,))
            gRq =  np.reshape(self.evaluationNV(NTq[:, q, :], N, I, params, params_names), (N,))
            
            #Obtain estimators
            V_sum = 0
            for n in range(N):
                V_sum = (g[n]**2 + gR[n]**2) + V_sum
            V = V_sum/2/N - f0**2
            
            gamma_sq_q_sum = 0
            for n in range(N):
                gamma_sq_q_sum = ((g[n]*gR[n]) + (gq[n]*gRq[n])) + gamma_sq_q_sum
            gamma_sq_q = gamma_sq_q_sum/2/N
            
            Vj = (np.dot(g, gq) + np.dot(gRq, gR))/2/N - gamma_sq_q
            
            Vmj = (np.dot(gRq, g) + np.dot(gq, gR))/2/N - gamma_sq_q
            
            #Obtain indices
            Sj = Vj/V
            STj = 1 - Vmj/V
            
            Sj_list_arr[q, :] = ([q, Sj])
            STj_list_arr[q, :] = ([q, STj])
    
        Sj_list_arr = sorted(Sj_list_arr, key = self.sortSecond, reverse = True)
        STj_list_arr = sorted(STj_list_arr, key = self.sortSecond, reverse = True)
        
        Sj_list_arr = np.array(Sj_list_arr)
        STj_list_arr = np.array(STj_list_arr)
        
        gsaV = np.column_stack((Sj_list_arr, STj_list_arr))
        
        return gsaV
    
    #Get n more important parameters
    def get_params(self, gsa, k, params_base):

        params_names = ['i0_an_H2_ref', 'i0_cat_ref', 'alpha_an_H2', 'alpha_cat', 'eta', 'B', 'Kappa_cat']
        
        paramsDict = {params_names[0]: 0, 
                      params_names[1]: 1, 
                      params_names[2]: 2, 
                      params_names[3]: 3, 
                      params_names[4]: 4, 
                      params_names[5]: 5, 
                      params_names[6]: 6}
        
        reverseparamsDict = {v: r for r, v in paramsDict.items()}
        
        params_valDict = {params_names[0]: params_base[0], 
                          params_names[1]: params_base[1], 
                          params_names[2]: params_base[2], 
                          params_names[3]: params_base[3], 
                          params_names[4]: params_base[4],
                          params_names[5]: params_base[5], 
                          params_names[6]: params_base[6]}
        
        ranked_params_names = [reverseparamsDict[item] for item in gsa[:, 0]]
        
        ranked_params_names= np.asarray(ranked_params_names[0:k])
        
        ranked_params_values= np.asarray([params_valDict[item] for item in ranked_params_names])
        
        return ranked_params_names, ranked_params_values
    
    #Plot results
    def plotgsa(self, gsa, params_names, I, SH2, SO2, T, P, IEC_mem, IEC_io, delta_mem, delta_io, CO_H2, L_c, base_params = [0.144, 2.63e-8, 1, 1.3, 4.8, 0.09, 2.8e5, 6000, 10000, 4000, 900, 1500, 650]):
        
        
        paramsDict = {params_names[0]: 0, 
                      params_names[1]: 1, 
                      params_names[2]: 2, 
                      params_names[3]: 3, 
                      params_names[4]: 4, 
                      params_names[5]: 5, 
                      params_names[6]: 6}
        reverseparamsDict = {v: k for k, v in paramsDict.items()}
        
        params_names_labels = ['$i_{0, an}^{ref}$', '$i_{0, cat}^{ref}$', '$\u03B1_{an}$', '$\u03B1_{cat}$', '\u03B3', 'B', '$\u039A_{cat}$']
        paramsDictLabels = {params_names_labels[0]: 0, 
                            params_names_labels[1]: 1, 
                            params_names_labels[2]: 2, 
                            params_names_labels[3]: 3, 
                            params_names_labels[4]: 4, 
                            params_names_labels[5]: 5, 
                            params_names_labels[6]: 6}
        
        reverseparamsDictLabels = {v: k for k, v in paramsDictLabels.items()}
        
        Sjs_labels = gsa[:, 0]
        Sjs_labels = [reverseparamsDictLabels[item] for item in Sjs_labels]
        Sjs_labels = np.array(Sjs_labels)
        
        STjs_labels = gsa[:, 2]
        STjs_labels = [reverseparamsDictLabels[item] for item in STjs_labels]
        STjs_labels = np.array(STjs_labels)
        
        fig = plt.figure(figsize = (15, 15))
        ax1 = fig.add_subplot(211)
        ax1.set_ylabel('$S_j$')
        ax1.bar(Sjs_labels[:], gsa[:, 1].astype(np.float), color = 'rebeccapurple', alpha = 0.6)
        
        ax2 = fig.add_subplot(212)
        ax2.set_ylabel('$ST_j$')
        ax2.bar(STjs_labels[:], gsa[:, 3].astype(np.float), color = 'rebeccapurple', alpha = 0.6)
        
        plt.rcParams['font.family']='sans-serif'
        tnfont={'fontname':'Helvetica'}
        plt.rcParams['font.size']=25
        plt.tick_params(direction = 'in')
        plt.tight_layout()
        plt.savefig('gsa.pdf')  
        plt.savefig('gsa.png')
        plt.show()
            
   
        self.operate(I, SH2, SO2, T, P, IEC_mem, IEC_io, delta_mem, delta_io, CO_H2, L_c, base_params, params_names, graphs = True)  
        
        return True
    
    #Plot results
    def plotgsaV(self, gsa, var_names, params_names, I, SH2, SO2, T, P, IEC_mem, IEC_io, delta_mem, delta_io, CO_H2, L_c, params = [0.144, 2.63e-8, 1, 1, 4.8, 0.09, 2.0e7]):
        
        
        varsDict = {var_names[0]: 0, 
                    var_names[1]: 1, 
                    var_names[2]: 2, 
                    var_names[3]: 3, 
                    var_names[4]: 4, 
                    var_names[5]: 5, 
                    var_names[6]: 6,
                    var_names[7]: 7, 
                    var_names[8]: 8, 
                    var_names[9]: 9}
        
        reversevarsDict = {v: k for k, v in varsDict.items()}
        
        var_names_labels = ['$SH_{2}}$', '$SO_{2}$', '$T$', '$P$', '$IEC_{mem}$', '$IEC_{io}$', '$\u03B4_{mem}$', '$\u03B4_{io}$', '$CO/H_{2}$', '$L_{c}$']
        varsDictLabels = {var_names_labels[0]: 0, 
                          var_names_labels[1]: 1, 
                          var_names_labels[2]: 2, 
                          var_names_labels[3]: 3, 
                          var_names_labels[4]: 4, 
                          var_names_labels[5]: 5, 
                          var_names_labels[6]: 6,
                          var_names_labels[7]: 7, 
                          var_names_labels[8]: 8, 
                          var_names_labels[9]: 9}
        
        reversevarsDictLabels = {v: k for k, v in varsDictLabels.items()}
        
        Sjs_labels = gsa[:, 0]
        Sjs_labels = [reversevarsDictLabels[item] for item in Sjs_labels]
        Sjs_labels = np.array(Sjs_labels)
        
        STjs_labels = gsa[:, 2]
        STjs_labels = [reversevarsDictLabels[item] for item in STjs_labels]
        STjs_labels = np.array(STjs_labels)
        
        fig = plt.figure(figsize = (20, 20))
        ax1 = fig.add_subplot(211)
        ax1.set_ylabel('$S_j$')
        ax1.bar(Sjs_labels[:], gsa[:, 1].astype(np.float), color = 'rebeccapurple', alpha = 0.6)
        
        ax2 = fig.add_subplot(212)
        ax2.set_ylabel('$ST_j$')
        ax2.bar(STjs_labels[:], gsa[:, 3].astype(np.float), color = 'rebeccapurple', alpha = 0.6)
        
        plt.rcParams['font.family']='sans-serif'
        tnfont={'fontname':'Helvetica'}
        plt.rcParams['font.size']=25
        plt.tick_params(direction = 'in')
        plt.tight_layout()
        plt.savefig('gsaV.pdf') 
        plt.savefig('gsaV.png') 
        plt.show()

        return True
    
    def jaya(self, M, M3, N, I, SH2, SO2, T, P, IEC_mem, IEC_io, delta_mem, delta_io, CO_H2, L_c, params, params_names, max_it, tol, r = 0.1):
        
        #The dimension, N, will change over iterations
        N = len(M[:, 0])
        
        g = self.evaluation(M, M3, N, I, SH2, SO2, T, P, IEC_mem, IEC_io, delta_mem, delta_io, CO_H2, L_c, params_names, jaya = True)
        
        #Add column of objective function outputs to array of solutions
        M = np.column_stack((M, g))
        
        #Drop nan values in M
        for n in range(N):
            if np.isnan(any(M[n, :])):
                M = np.delete(M, obj = n, axis = 0)
        
        #Sort from best to worst
        M = np.asarray(sorted(M, key = self.sortLast, reverse = False))
        
        #Get the best solution
        best = M[0, :]
        
        #Get the worst solution
        worst = M[-1, :]
        
        it = 0
        
        while it < max_it:
            for n in range(len(M[:, 0])):
                #Update only the k highest ranked parameters
                for i in range(len(params_names)):

                    M[n, i] = M[n, i] + random.uniform(0, r)*(best[i] - abs(M[n, i])) - random.uniform(0, r)*(worst[i] - abs(M[n, i]))
            
            g = self.evaluation(M, M3, len(M[:, 0]), I, SH2, SO2, T, P, IEC_mem, IEC_io, delta_mem, delta_io, CO_H2, L_c, params_names, jaya = True)

            #Drop previous iteration
            #Get new iteration
            M = np.column_stack((M, g))
            #Drop zero columns
            M = M[M[:, -1] != 0]
            
            #Drop unfeasible solutions
            k = 0
            while k < len(M[:, 0]):
                if M[k, -1] > best[-1]:
                    M = np.delete(M, obj = k, axis = 0)
                k += 1
                
            M = np.asarray(sorted(M, key = self.sortLast, reverse = False))

            best = M[0, :]
            worst = M[-1, :]
            
            if np.max(best[-1]) < tol:
                break
            
            it += 1
        
        return best
    
    def data_gen(self, p_size, s_size, params, params_names, 
                 interval = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], 
                 base_values = ([1.2, 2.2, 493, 1.59, 2.2, 1.72, 0.005, 0.0001, 0.1, 0.3])):
        
        A = 5
        I = np.linspace(0.01, 24, 100)

        values_up = np.zeros(len(base_values), )
        values_low = np.zeros(len(base_values), )
        
        for i in range(len(base_values)):
            value_up = base_values[i] +base_values[i]*interval[i]
            values_up[i] = value_up
            value_low = base_values[i] - base_values[i]*interval[i]
            if value_low < 0:
                value_low = 0
            values_low[i] = value_low        
        
        SH2_pop             = np.linspace(values_low[0], values_up[0], p_size).tolist()
        SO2_pop             = np.linspace(values_low[1], values_up[1], p_size).tolist()
        T_pop               = np.linspace(values_low[2], values_up[2], p_size).tolist()
        P_pop               = np.linspace(values_low[3], values_up[3], p_size).tolist()
        IEC_pop             = np.linspace(values_low[4], values_up[4], p_size).tolist()
        IEC_io_pop           = np.linspace(values_low[5], values_up[5], p_size).tolist()
        delta_mem_pop       = np.linspace(values_low[6], values_up[6], p_size).tolist()
        delta_io_pop        = np.linspace(values_low[7], values_up[7], p_size).tolist()
        CO_H2_pop           = np.linspace(values_low[8], values_up[8], p_size).tolist()
        L_c_pop             = np.linspace(values_low[9], values_up[9], p_size).tolist()
        
        SH2_sample          = random.sample(SH2_pop, s_size)
        SO2_sample          = random.sample(SO2_pop, s_size)
        T_sample            = random.sample(T_pop, s_size)
        P_sample            = random.sample(P_pop, s_size)
        IEC_sample          = random.sample(IEC_pop, s_size)
        IEC_io_sample        = random.sample(IEC_io_pop, s_size)
        delta_mem_sample    = random.sample(delta_mem_pop, s_size)
        delta_io_sample     = random.sample(delta_io_pop, s_size)
        CO_H2_sample        = random.sample(CO_H2_pop, s_size)
        L_c_sample          = random.sample(L_c_pop, s_size)
        
        c = []
        for j in range(s_size):
            c_j = [SH2_sample[j], SO2_sample[j], T_sample[j], 
                   P_sample[j], IEC_sample[j], IEC_io_sample[j], 
                   delta_mem_sample[j], delta_io_sample[j] , 
                   CO_H2_sample[j], L_c_sample[j]]
            c.append(c_j)
        c = np.asarray(c)

        power = []
        polCurves = []
        for jx in range(len(c)):
            overpotentials_jx = self.operate(I, 
                                             c[jx, 0], c[jx, 1], c[jx, 2], c[jx, 3], 
                                             c[jx, 4], c[jx, 5], c[jx, 6], c[jx, 7], 
                                             c[jx, 8], c[jx, 9], params, params_names)
            
            peak_power = max(overpotentials_jx[1]*np.reshape(I, (len(I),))/A)
            
            power.append(random.gauss(peak_power, self.f0))
            polCurves.append(overpotentials_jx[1])           

        power = np.asarray(power)
        
        samples = np.column_stack([c, power])
        names = ['SH2', 'SO2', 'T', 'P', 'IEC', 'IEC_io', 'delta_mem', 'delta_io','CO_H2', 'L_c', 'Power']
        FuelCellModelData = pd.DataFrame(samples, columns = names)
        FuelCellModelData.dropna()
        FuelCellModelData = FuelCellModelData[(FuelCellModelData.T != 0).any()]
        today = date.today()
        FuelCellModelData.to_excel(f'DataSample {s_size} points {today}.xlsx', index=True, header=True)
        return polCurves