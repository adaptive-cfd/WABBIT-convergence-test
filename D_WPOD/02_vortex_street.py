#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
##############################################
WAVELET PROPPER ORTHOGONAL DECOMPOSITION

MAIN skript calling all subroutines vortex street
##############################################
"""
from wPOD.run_wabbit import run_wabbit_POD
import numpy as np
from wPOD.wPODerror import *
###############################################################################
###############################################################################
# directories needed
dirs["work"]+="/cyl/"
dirs["images"]+="/cyl/"

data = {"folder" :  "/home/phil/develop/WPOD/D_WPOD/cyl/", # where the data is stored
        "qname" : ["p"]
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
Jmax_list = [6]
Jmax_dir_list = [ "Jmax"+str(Jmax)+"/" for Jmax in Jmax_list]

eps_list = np.asarray([0]+[float("%1.1e" %eps) for eps in np.logspace(-5,0,14)])

eps_dir_list = [ "eps%1.1e" %eps for eps in eps_list]


mode_lists = ["mode1_list.txt","mode2_list.txt","mode3_list.txt","mode4_list.txt"]
reconstructed_iteration =7

run_wabbit_POD(wabbit_setup, dirs, data, Jmax_list, eps_list, mode_lists, reconstructed_iteration)



###############################################################################
###############################################################################
# %% 
delta_err,clist=plot_wPODerror(Jmax_list, Jmax_dir_list,eps_dir_list,eps_list, dirs)
