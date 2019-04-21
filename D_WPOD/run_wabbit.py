#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 11:16:19 2019

@author: Philipp Krah

This script runs wabbit-post --POD for different parameters of the 
threshold value eps.
The task is to study the dependence of the adaptation on the DECOMPOSITION.

"""

import re
import os
import glob

# %% Change parameters here:
eps_list = [100, 10, 1, 0.1, 0.01, 0.001]
wdir = "/home/phil/devel/WABBIT/"                                              # directory of wabbit-post 
data = "/home/phil/devel/results/cyl_POD/wPOD/vor_crop_adapt/vorx_0000*.h5"  # data you want to use for POD
mpicommand = "mpirun -np 4"                                                    # mpi command used for parallel execution
memory ="--memory=8GB"                                                         # how much memory (RAM) available on your pc
command = mpicommand + " " +  wdir + \
         "wabbit-post --POD --save_all --nmodes=30 " + memory + \
         " --adapt=%s " + data

for eps in eps_list:
    # -----------------------------
    # Prepare to save data from POD
    # -----------------------------
    c = command % str(eps)   # change eps
    c += " > wPOD.log" 
    save_dir = "eps" + str(eps)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir) # make directory for saving the files
    os.chdir(save_dir)
    
    # -----------------------------
    # Execute Command
    # -----------------------------
    print("\n\n###################################################################")
    print("\t\teps =",eps)
    print("###################################################################")
    print("\n",c,"\n\n")
    success = os.system(c)   # execute command
    os.chdir("../")
    if success != 0:
        print("command did not execute successfully")
        break
    