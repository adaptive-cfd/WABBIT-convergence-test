#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
##############################################
WAVELET PROPPER ORTHOGONAL DECOMPOSITION

MAIN skript calling all bumblebee routines
##############################################
"""
from wPOD.run_wabbit import run_wabbit_POD
import numpy as np
from wPOD.wPODerror import *
from wPODdirs import *
###############################################################################
###############################################################################
# %% directories needed
# directories needed
dirs["work"]+="/bumblebee/"
dirs["images"]+="/bumblebee/"
# setup for wabbit call
wabbit_setup["memory"] = "--memory=700GB"

data = {"folder" :  "/work/krah/bumblebee/POD/",
        "qname" : ["ux", "uy", "uz", "p"]
        }

###############################################################################
###############################################################################
# %% In this step we call all necessary wabbit routines
#           * --POD
#           * --POD-reconstruct
#           * --PODerror 
#  for the given parameters
###############################################################################

Jmax_list = [7]
Jmax_dir_list = [ "Jmax"+str(Jmax)+"/" for Jmax in Jmax_list]

eps_list = np.asarray([float("%1.1e" %eps) for eps in np.logspace(-2,0,5)])
eps_dir_list = [ "eps%1.1e"%(eps) for eps in eps_list]

reconstructed_iteration = 7
mode_lists = ["mode1_list.txt","mode2_list.txt","mode3_list.txt","mode4_list.txt"]

#run_wabbit_POD(wabbit_setup, dirs, data, Jmax_list, eps_list, mode_lists, reconstructed_iteration)

###############################################################################
###############################################################################
# %% In this step we analyze the results
#  for the given parameters
###############################################################################
# %% 
plot_wPODerror(Jmax_list, Jmax_dir_list,eps_dir_list,eps_list, dirs, eps_list_plot=np.delete(eps_list,[1,2]),show_legend=True, alternate_markers=True)
