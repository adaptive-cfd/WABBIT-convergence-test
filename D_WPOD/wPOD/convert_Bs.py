#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 12:08:33 2020

@author: krah
"""

from wPOD.run_wabbit import run_wabbit_POD, compute_vorticity
import numpy as np
import wabbit_tools as wt
from wPOD import adapt2eps
from wPODdirs import *
from wPOD.wPODerror import *
from wPOD.sci_notation import sci_notation
from wPOD import plot_vector_modes
from matplotlib.offsetbox import AnchoredText
import os
import farge_colormaps
from matplotlib import rc

dir_list={}
dir_list["work"]=dirs["work"] +"/cyl/"
dir_list["images"]= dirs["images"] + "/cyl/"
dir_list["wabbit"]= wdir

data = {"folder" :  home+"/develop/results/20200908_cylinder/", # where the data is stored
        "qname" : ["ux", "uy", "p"]
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
eps_list = np.asarray([0]+[float("%1.1e" %eps) for eps in np.logspace(-5,0,10)])
eps_list = np.delete(eps_list,2)
#eps_list = np.asarray([0]+[float("%1.1e" %eps) for eps in np.logspace(-7,0,10)])
eps_dir_list = [ "eps%1.1e" %eps for eps in eps_list]
jmax=Jmax_list[0]

mode_lists = ["mode1_list.txt","mode2_list.txt","mode3_list.txt","mode4_list.txt"]
reconstructed_iteration =7


    # read data
files=sorted(glob.glob(data["folder"]+"Jmax"+str(jmax)+"/u*.h5"))[275:]
p_list=[]
t_list=[]
for fname_in in files:

    path,fname = os.path.split(fname_in)
    folder = data['folder'] + "/Jmax"+str(jmax-1)+"/"
    fname_out= folder+fname.split("_")[0]
    time, x0, dx, box, datafield, treecode = wt.read_wabbit_hdf5( fname_in )
    field, box = wt.dense_matrix(  x0, dx, datafield, treecode)
    p_list.append(field[786,2048])
    t_list.append(time)
    field = np.squeeze(field)
    Bs = wt.field_shape_to_bs(field.shape,jmax)
    box = box[::-1]
    fname=wt.dense_to_wabbit_hdf5(field, fname_out , Bs, box, time)
    
#    plt.figure(55)
#    plt.pcolormesh(field)
#    plt.axis("image")
#    plt.xticks([])
#    plt.yticks([])
#    plt.savefig(fname.replace("h5","png"))
    
plt.figure(3)
plt.plot(t_list, p_list)
