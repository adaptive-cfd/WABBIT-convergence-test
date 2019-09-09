#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
##############################################
WAVELET PROPPER ORTHOGONAL DECOMPOSITION

MAIN skript calling all subroutines
##############################################
"""
from run_wabbit import run_wabbit_POD
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
""" 
 Vortex Street
"""
exponent = np.arange(-5,1)

Jmax_list = [4, 5, 6]
Jmax_dir_list = [ "Jmax"+str(Jmax)+"/" for Jmax in Jmax_list]

eps_list = [np.round(val,decimals=-exp) for exp in exponent for val in np.arange(4,11,4)*10.0**(exp)]
eps_dir_list = [ "eps"+str(eps) for eps in eps_list]

#run_wabbit_POD(wabbit_setup, dirs, data, Jmax_list, eps_list, mode_lists, reconstructed_iteration)


"""
 BumbleBee
"""
Jmax_list = [5]
eps_list = [1.6 *1e-2, 1e-2, 1e-1,0.5]
reconstructed_iteration = 7
data = {"folder" :  "/home/krah/develop/results/bumblebee",
        "qname" : "vorabs"
        }
mode_lists = ["mode1_list.txt"]

run_wabbit_POD(wabbit_setup, dirs, data, Jmax_list, eps_list, mode_lists, reconstructed_iteration)

###############################################################################
###############################################################################
# %% 