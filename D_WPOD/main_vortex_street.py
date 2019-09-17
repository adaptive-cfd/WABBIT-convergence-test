#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
##############################################
WAVELET PROPPER ORTHOGONAL DECOMPOSITION

MAIN skript calling all subroutines vortex street
##############################################
"""
from run_wabbit import run_wabbit_POD
import numpy as np
from wPODerror import *
###############################################################################
###############################################################################
# %% directories needed
dirs= {
       'wabbit' : "~/develop/WABBIT/" ,         # directory of wabbit
       'work'   : "./",                         # where to save big data files
       'images' : "./images"                           #pictures are saved here
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
""" 
 Vortex Street
"""
exponent = np.arange(-5,1)

Jmax_list = [4, 5]
Jmax_dir_list = [ "Jmax"+str(Jmax)+"/" for Jmax in Jmax_list]

eps_list = [np.round(val,decimals=-exp) for exp in exponent for val in np.arange(4,7,4)*10.0**(exp)]
eps_dir_list = [ "eps"+str(eps) for eps in eps_list]

#run_wabbit_POD(wabbit_setup, dirs, data, Jmax_list, eps_list, mode_lists, reconstructed_iteration)



###############################################################################
###############################################################################
# %% 
 plot_wPODerror(Jmax_list, Jmax_dir_list,eps_dir_list,eps_list, dirs)