#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 10:55:09 2019

@author: krah
"""
import numpy as np
import wabbit_tools as wt
import matplotlib.pyplot as plt
import re
import os
import glob
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d

###############################################################################
# %% Change parameters here
###############################################################################
# directories needed
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
# parameters to adjust
class params:
    
    jmax_list = np.arange(1,10)        # maximal tree level
    predictor_order= 4
    class domain:
        N = [16, 16]            # number of points for 3d use 3 elements
        L = [1, 1]              # length of domain
        
    def init(N):                             # this function will be initialized on the domain
        field = np.zeros(N)
        field[N[0]//2,N[1]//2]=1
        return field     
###############################################################################
# %% 
###############################################################################



def wabbit_adapt(dirs, params, wabbit_setup):
    
    
    mpicommand = wabbit_setup["mpicommand"]
    memory     = wabbit_setup["memory"]
    wdir = dirs["wabbit"]
    work = dirs["work"]
    jmax_list = params.jmax_list
    # set initial gird
    dim = len(params.domain.L)
    x = [np.linspace(0,params.domain.L[d],params.domain.N[d]) \
         for d in range(dim) ]
    X = np.meshgrid(*x)
    phi = params.init(params.domain.N)
    Bs = wt.field_shape_to_bs(params.domain.N,jmax_list[0])
    
    # create reference file
    file_coarse = wt.dense_to_wabbit_hdf5(phi,work+"/phi", Bs, params.domain.L,0)
    
    for j,jmax in enumerate(params.jmax_list):

        print("\n\n###################################################################")
        print("###################################################################")
        print( "\t\tJmax: ", jmax, "\n\n")
        fname = "phi-j"+str(jmax)+".h5"
        file_dense = work +"/"+fname
        # command for densing file again
        dense_command = mpicommand + " " +  wdir + \
                 "wabbit-post"+ " --sparse-to-dense " + \
                  " "+ file_coarse + " "+file_dense + " "+str(jmax)+" "+ \
                  str(params.predictor_order)+ " >adapt.log"
        # -----------------------------
        # Execute Command
        # -----------------------------
        print("\n",dense_command,"\n\n")
        success = os.system(dense_command)   # execute command
        #wt.plot_wabbit_file(file_dense,savepng=True)
        if success != 0:
                print("command did not execute successfully")
                return
            
    # delete reference file
    os.system("rm " + file_coarse)
    return file_dense

file = wabbit_adapt(dirs, params, wabbit_setup)        

# Grab some test data.
time, x0, dx, box, field, treecode = wt.read_wabbit_hdf5(file)
field, box = wt.dense_matrix(x0,dx,field,treecode)

# %%
plt.close("all")

fig1 = plt.figure()
ax1 = fig1.gca(projection='3d')
Nxy=field.shape
Z = field[Nxy[0]//3:-Nxy[0]//3:40, Nxy[1]//3:-Nxy[1]//3:40]

x = np.linspace(0,params.domain.L[0],Z.shape[0])
y = np.linspace(0,params.domain.L[1],Z.shape[1])
[X,Y] = np.meshgrid(x,y)
# Plot a basic wireframe.


ax1.plot_wireframe(X,Y, Z, rstride=1, cstride=1,  antialiased = True,
                   color = "black")

ax1.set_axis_off()
# Equally stretch all axes
ax1.set_aspect("equal")
plt.show()


#
################################################
#%% save figure
###############################################
fig1.savefig( dirs["images"]+'waveletDD'+str(params.predictor_order)+'.png', dpi=300, transparent=True, bbox_inches='tight' )# -*- coding: utf-8 -