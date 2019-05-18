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
import numpy as np
from wPODdirs import *
# %% Change parameters here:
#eps_list = [100, 10, 1, 0.1, 0.01, 0.001]
eps_list = [0, 0.001, 0.01]

wdir = "/home/phil/devel/WABBIT/"                                              # directory of wabbit-post 
data = "/home/phil/devel/results/cyl_POD/wPOD/vor_crop_adapt/vorx_0000*.h5"    # data you want to use for POD
mpicommand = "mpirun -np 4"                                                    # mpi command used for parallel execution
memory ="--memory=16GB"                                                         # how much memory (RAM) available on your pc



# %% run wPOD for different eps:
def run_wPOD_for_different_eps(wdir, data, eps_list, memory, mpicommand, save_log=False):
    
   
    for eps in eps_list:
        # ------------------------------
        # BUILD EXECUTION COMMAND
        # ------------------------------
        if eps > 0: 
            c = mpicommand + " " +  wdir + \
             "wabbit-post --POD --save_all --nmodes=30 " + memory + \
             " --adapt="+ str(eps) + " " + data
             
        else:
            c = mpicommand + " " +  wdir + \
             "wabbit-post --POD --save_all --nmodes=30 " + memory + \
             " " + data
        # pipe output into logfile     
        if save_log:
            c += " > wPOD.log" 
        # -----------------------------
        # Prepare to save data from POD
        # -----------------------------
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
    return success

# %% reconstruct for different eps:
def run_wPOD_reconstruction_for_different_eps(wdir, data, eps_list, memory, mpicommand):    
    command = mpicommand + " " +  wdir + \
             "wabbit-post --POD-reconstruct --save_all --nmodes=30 " + memory + \
             " --adapt=%s " + data
    for eps in eps_list:
        # -----------------------------
        # Prepare to save data from POD
        # -----------------------------
        c = command % str(eps)   # change eps
        c += " > reconstruct.log" 
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
        return success

# %% adapt one snapshot for different eps: logfile written in adapt .log
def adpatation_for_different_eps(wdir, h5_filename, eps_list, memory, mpicommand, create_log_file=True):
    file = data_file.split("/")[-1]
    command = mpicommand + " " +  wdir + \
             "wabbit-post --dense-to-sparse --eps=%s " + memory + \
              " "+ file
              
    for eps in eps_list:
        # -----------------------------
        # Prepare to save data from POD
        # -----------------------------
        c = command % str(eps)   # change eps
        if create_log_file:
            c += " > adapt.log" 
        save_dir = "eps" + str(eps)
        if not os.path.exists(save_dir):
            os.mkdir(save_dir) # make directory for saving the files
        os.chdir(save_dir)
        # copy given file in current directory 
        os.system("cp "+h5_filename+ " .")
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
    return success

# %% run scripts:
run_wPOD_for_different_eps(wdir, data, eps_list, memory, mpicommand, save_log=True)
#run_wPOD_for_different_eps(wdir, data, eps_list, save_dir, memory, mpicommand)
#run_wPOD_reconstruction_for_different_eps(wdir, data, eps_list, memory, mpicommand)
#data_file = np.sort(glob.glob(data))[100]
#adpatation_for_different_eps(wdir, data_file, eps_list, memory, mpicommand)