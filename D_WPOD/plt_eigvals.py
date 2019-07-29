#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 12:36:33 2019

@author: Philipp Krah

Tasks of this skript:
    * Plot eigenvalues for different eps
"""
###############################################################################
#           MODULES
###############################################################################
import os
#import farge_colormaps
import glob
from wPODdirs import *
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc
###############################################################################
# LATEX FONT
font = {'family' : 'serif',
        'size'   : 18}
rc('font',**font)
rc('text', usetex=True)   
###############################################################################
###############################################################################
# Specify:

#eps_list = [100, 10, 1, 0.1, 0.01, 0.001, 0]
#eps_list = [ 0.1,  0]
eps_dir_list = [ "eps"+str(eps) for eps in eps_list]
###############################################################################
# %% get the data
plt.close("all")
eigval_list={}
fig, ax = plt.subplots()
markers = ['o', 'x', '+', 'v', '^', '<', '>', 's', 'd']
for i, eps_dir in enumerate(eps_dir_list):
    eigvals_file = eps_dir + "/eigenvalues.txt"
    ########################
    # get data from file
    #######################
    f = open(eigvals_file)
    data_eigs = [[np.sqrt(float(num)) for num in line.split()] for line in f]
    data_eigs = np.asarray(data_eigs)
    f.close
     ########################
    # save in dictionary
    #######################
    key = eps_list[i]
    eigval_list.update({ key : data_eigs})
    ########################
    # Do the actual plotting
    #######################
    if eps_list[i]==0:
        my_label= '$\epsilon ='+str(key) +'$ (dense)'
    else:
        my_label= '$\epsilon ='+str(key) +'$ (sparse)'
        
    ax.plot(data_eigs[::-1,1], linestyle='',marker = markers[i], mfc='none',\
                markersize=5,label=my_label)

ax.legend(loc='upper right')
ax.set_yscale('log')
plt.xlabel("Mode number $k$", fontsize=24)
plt.ylabel("$\sigma^\epsilon_k$",fontsize=24)
plt.savefig( pic_dir+"eigvals.eps", dpi=300, transparent=True, bbox_inches='tight' )
fig.show()
