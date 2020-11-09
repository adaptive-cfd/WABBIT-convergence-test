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
from wPOD.run_wabbit import compute_vorticity
import os
from os import path
from wPOD.adaption_test import adapt2eps

###############################################################################
###############################################################################
# %% directories needed
# directories needed
dir_list = dirs.copy()
dir_list["work"]=dirs["work"]+"/bumblebee/"
dir_list["images"]=dirs["images"]+"/bumblebee/"
# setup for wabbit call
wabbit_setup["memory"] = "--memory=100GB"

data = {"folder" :  "/home/phil/develop/results/bumblebee/",
        "qname" : ["vorx", "vory", "vorz"]
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

eps_list = np.asarray([0]+[float("%1.1e" %eps) for eps in np.logspace(-2,0,4)])
eps_dir_list = [ "eps%1.1e"%(eps) for eps in eps_list]

reconstructed_iteration = 7
mode_lists = ["mode1_list.txt","mode2_list.txt","mode3_list.txt","mode4_list.txt"]

# %% Compression ERROR
eps_list = np.asarray([float("%1.1e" %eps) for eps in np.logspace(-2,0,4)])
wdir = dirs["wabbit"]
fig_adapt = [plt.figure(41)]
ax_adapt = [fig.add_subplot() for fig in fig_adapt]
h5_fname = [sorted(glob.glob(data["folder"]+'/Jmax7/'+qty+'*25*.h5'))[0] for qty in data["qname"]]
[l2error, linferror, Nblocks, Nblocksdense]=adapt2eps(h5_fname, \
wdir, eps_list, wabbit_setup['memory'], wabbit_setup['mpicommand'], wavelets="CDF44",\
normalization = "L2",create_log_file=True, show_plot=True, pic_dir=dir_list["images"])


    # %% Compute vorticity if not allready exists

ux_files=sorted(glob.glob(data["folder"]+"CDF44_jmax"+str(Jmax_list[0])+'/ux_*.h5'))
uy_files=sorted(glob.glob(data["folder"]+"CDF44_jmax"+str(Jmax_list[0])+'/uy_*.h5'))
uz_files=sorted(glob.glob(data["folder"]+"CDF44_jmax"+str(Jmax_list[0])+'/uz_*.h5'))

for fux,fuy,fuz in zip(ux_files,uy_files,uz_files):
    
    fvorx = fux.split("/")[-1].replace('ux_','vorx_')
    fvory = fuy.split("/")[-1].replace('uy_','vory_')
    fvorz = fuz.split("/")[-1].replace('uz_','vorz_')
    folder = os.path.split(fux)[0]

    if not path.exists(path.join(folder,fvorx)):

        success=compute_vorticity(wdir, fux, fuy, wabbit_setup["memory"], wabbit_setup["mpicommand"],uzfile=fuz)

        os.system("mv "+fvorx+" "+ folder)
        os.system("mv "+fvory+" "+ folder)
        os.system("mv "+fvorz+" "+ folder)



###############################################################################
###############################################################################
# %% In this step we analyze the results
#  for the given parameters
###############################################################################
# %% 
delta_err,clist=plot_wPODerror(Jmax_list, Jmax_dir_list,eps_dir_list,eps_list, \
                               dir_list,eps_list_plot=eps_list, alternate_markers=True, \
                               show_legend=True, ref_eigvals=True, n_star=30)