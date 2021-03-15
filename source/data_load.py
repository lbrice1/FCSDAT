# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 17:41:54 2020

@author: lbrice1
"""

import os
import sys
import os.path
import pandas as pd

#Set working source as working directory
os.chdir('./source')
def data_load(filename):
    os.chdir('..')
    d = os.getcwd()
    o = [os.path.join(d, o) for o in os.listdir(d) if os.path.isdir(os.path.join(d, o))] 
    for item in o:
        if os.path.exists(item + filename):
            file = item + filename
    data = pd.read_excel(file)
    os.chdir('./source')
    return data
