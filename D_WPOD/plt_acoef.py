#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 12:43:01 2019

@author: phil

Tasks of this skript:
    * Plot acoef for different eps
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
        'size'   : 16}
rc('font',**font)
rc('text', usetex=True)   
###############################################################################
###############################################################################
# Specify:

eps_list = [100, 10, 1, 0.1, 0.01, 0.001, 0]
eps_dir_list = [ "eps"+str(eps) for eps in eps_list]
mode =0
###############################################################################
# %% get the data
plt.close("all")
eigval_list={}
fig, ax = plt.subplots()
markers = ['o', '.', 'x', '+', 'v', '^', '<', '>', 's', 'd']
for i, eps_dir in enumerate(eps_dir_list):
    file = eps_dir + "/a_coefs.txt"
    ########################
    # get data from file
    #######################
    f = open(file)
    data = [[float(num) for num in line.split()] for line in f]
    data = np.asarray(data)
    f.close
     ########################
    # save in dictionary
    #######################
    key = eps_list[i]
    eigval_list.update({ key : data})
    ########################
    # Do the actual plotting
    #######################
    ax.plot(data[:,mode], linestyle='-',marker = markers[i], mfc='none',\
                markersize=5,label='$\epsilon ='+str(key) +'$')

ax.legend(loc='upper right')
plt.xlabel("$t$")
plt.ylabel("$a_{"+str(mode)+"}(t)$")
plt.savefig( pic_dir+"acoef.eps", dpi=300, transparent=True, bbox_inches='tight' )
fig.show()
