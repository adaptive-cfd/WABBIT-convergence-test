#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 13:41:25 2019

@author: phil
"""


import wabbit_tools as wt
import matplotlib.image as img 
import matplotlib.pyplot as plt
import flusi_tools as flusi
import numpy as np
import insect_tools
import os
import farge_colormaps
import glob
from matplotlib import rc
font = {'family' : 'sans-serif',
        'sans-serif':['Helvetica'],
        'size'   : 22}
rc('font',**font)
rc('text', usetex=True)     


pic_dir = "/home/phil/devel/WABBIT-convergence-test/D_WPOD/images/"
eps_dir_list = [ "eps"+str(eps) for eps in eps_list]
#wdir = "/home/krah/develop/WABBIT/"
wdir = "~/savart/develop/WABBIT/"
resdir="/home/phil/devel/results/cyl_POD/"

fc = farge_colormaps.farge_colormap_multi( taille=600,limite_faible_fort=0.2, etalement_du_zero=0.04, type='vorticity' )





# %% Plot flusi original files:
files = glob.glob(resdir+'wPOD/reconstruct/*1.h5')
files.sort()

fig = plt.figure()
Nxcut = [50, 206]
Nycut = [128, 128]
Nt = len(files)
Npics = 2
iter=0
step=1#Nt//Npics
for iter,file in enumerate(files[::step]):
        flusi.crop_flusi_HDF5(file,Nxcut,Nycut)

# %% convert and adapt
wt.flusi_to_wabbit_dir(resdir+'wPOD/reconstruct/', resdir+'wPOD/reconstruct/',level=4 )
#wt.command_on_each_hdf5_file("/home/phil/devel/results/cyl_POD/wPOD/reconstruct/",\
#                            "/home/phil/devel/WABBIT/wabbit-post --dense-to-sparse --eps=0.1 %s")
# %% Plot flusi original files:        
files = glob.glob(resdir+'/wPOD/reconstruct/'+'/*.h5')
files.sort()
Nt=len(files)
Npics=Nt
step=Nt//Npics
for iter,file in enumerate(files[::step]):
    plt_file = file.split("/")[-1]
    plt_file = plt_file.replace('h5','png')
    fig, ax = plt.subplots() 
    ax,cb = wt.plot_wabbit_file(file,cmap=fc,caxis_symmetric=True,dpi=300,\
                                shading='gouraud',caxis=[-10,10],block_edge_alpha=0.2,\
                                ticks=False, \
                                colorbar_orientation="horizontal",title=False, colorbar=False)
   # ax.set_title("Snapshot $u^\epsilon(\mathbf{x},t_i)$ at $i="+str(iter*step)+"$")
    #ax.set_xlabel("$x$")
    #ax.set_ylabel("$y$")
    ax.set_aspect('equal')
    #cb.set_label("vorticity [1/s]")
    plt.savefig( pic_dir+plt_file, dpi=300, transparent=True, bbox_inches='tight' )
    plt.close()
    
# %%
