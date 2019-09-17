#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
##############################################
WAVELET PROPPER ORTHOGONAL DECOMPOSITION

MAIN skript calling all bumblebee routines
##############################################
"""
from run_wabbit import run_wabbit_POD
import numpy as np
###############################################################################
###############################################################################
# %% directories needed
dirs= {
       'wabbit' : "~/develop/WABBIT/" ,         # directory of wabbit
       'work'   : "./",                         # where to save big data files
       'images' : "./"                           #pictures are saved here
       }
# setup for wabbit call
wabbit_setup = {
        'mpicommand' : "mpirun -np 4",
        'memory'     : "--memory=32GB"
        }
###############################################################################
###############################################################################
# %% In this step we call all necessary wabbit routines
#           * --POD
#           * --POD-reconstruct
#           * --PODerror 
#  for the given parameters
###############################################################################
Jmax_list = [6]
eps_list = [4.0 *1e-2, 8.0e-2, 4.0e-1,0.8]
Jmax_dir_list = [ "../../WABBIT-convergence-bumblebee//D_WPOD/Jmax"+str(Jmax)+"/" for Jmax in Jmax_list]
eps_dir_list = [ "eps"+str(eps) for eps in eps_list]

reconstructed_iteration = 7
data = {"folder" :  "/home/krah/develop/results/bumblebee/",
        "qname" : ["ux", "uy", "uz", "p"]
        }
mode_lists = ["mode1_list.txt","mode2_list.txt","mode3_list.txt","mode4_list.txt"]

run_wabbit_POD(wabbit_setup, dirs, data, Jmax_list, eps_list, mode_lists, reconstructed_iteration)


###############################################################################
###############################################################################
# %% In this step we analyze the results
#  for the given parameters
###############################################################################
# %% 
 plot_wPODerror(Jmax_list, Jmax_dir_list,eps_dir_list,eps_list, dirs)
