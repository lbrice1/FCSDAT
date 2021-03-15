# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 14:35:26 2020

@author: lbrice1
"""

import numpy as np
from numpy import sqrt, exp, log
import matplotlib.pyplot as plt


#data
labels = ['$SO_{2}$', '$IEC_{io}$', '$T$', '$CO/H_{2}$', '$IEC_{mem}$', '$L_{c}$','$SH_{2}}$', '$P$', '$\u03B4_{mem}$']
frequency = [12, 9, 8, 7, 6, 2, 1, 1, 1]


fig = plt.figure(figsize = (20, 15))

ax1 = fig.add_subplot(211)
ax1.set_ylabel('Frequency')
ax1.bar(labels[:], frequency[:], color = 'teal', alpha = 0.6)
ax1.text(0.006, 0.98, '(a)', fontsize = 'x-large', horizontalalignment='left',verticalalignment='top', transform=ax1.transAxes)
ax1.legend(['All clusters'])

labels2 = ['$IEC_{io}$', '$SO_{2}$', '$T$', '$CO/H_{2}$', '$P$', '$IEC_{mem}$', '$L_{c}$','$SH_{2}}$', '$\u03B4_{mem}$']
frequency2 = [3, 2, 1, 1, 1, 0, 0, 0, 0]


ax2 = fig.add_subplot(212)
ax2.set_ylabel('Frequency')
ax2.bar(labels2[:], frequency2[:], color = 'rebeccapurple', alpha = 0.6)
ax2.text(0.006, -0.18, '(b)', fontsize = 'x-large', horizontalalignment='left',verticalalignment='top', transform=ax1.transAxes)
ax2.legend(['Cluster 5'])

plt.rcParams['font.family']='sans-serif'
tnfont={'fontname':'Helvetica'}
plt.rcParams['font.size']=25
plt.tick_params(direction = 'in')
plt.tight_layout()
plt.savefig('../figures/SGScluster5.pdf') 
plt.savefig('../figures/SGScluster5.png') 
plt.show()
