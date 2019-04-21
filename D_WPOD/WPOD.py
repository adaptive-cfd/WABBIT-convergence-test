#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 12:16:23 2019

@author: krah

 This script generates all post processing and plots for analyzing 
 the 
 
         WAVELET ADAPTIVE PROPER ORTHOGONAL DECOMPOSITION (wPOD)
         

         
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


pic_dir = "./images/"
#wdir = "/home/krah/develop/WABBIT/"
wdir = "~/savart/develop/WABBIT/"
resdir_flusi="../results/cyl_POD/wPOD/vor_crop/"
resdir_flusi_modes="../results/cyl_POD/vorticity_POD2/modes/"
resdir_wPOD_modes="~/savart/develop/WABBIT/19.04.19/"
resdir_wabbit =resdir_flusi + "_adapt/"

fc = farge_colormaps.farge_colormap_multi( taille=600,limite_faible_fort=0.2, etalement_du_zero=0.04, type='vorticity' )





# %% Plot flusi original files:
files = glob.glob(resdir_flusi+'/*.h5')
files.sort()

fig = plt.figure()
Nxcut = [50, 206]
Nycut = [128, 128]
Nt = len(files)
Npics = 2
iter=0
step=1#Nt//Npics
for iter,file in enumerate(files[::step]):
       # flusi.crop_flusi_HDF5(file,Nxcut,Nycut)
        plt_file = file.split("/")[-1]
        plt_file = plt_file.replace('h5','png')
        time, box, origin, data_flusi = insect_tools.read_flusi_HDF5( file )
        data = np.squeeze(data_flusi).T
        y = np.linspace(0,box[-1],np.size(data,0))
        x = np.linspace(0,box[-2],np.size(data,1))
        X,Y = np.meshgrid(x,y)
        plt.pcolormesh(X,Y,data, cmap=fc)#'RdBu')
        plt.title("Snapshot $u(\mathbf{x},t_i)$ at $i="+str(iter*step)+"$")
        plt.clim(-10,10)
        cl=plt.colorbar(orientation="horizontal")
        cl.set_label("vorticity [1/s]")
        plt.xlabel("$x$")
        plt.ylabel("$y$")
        plt.axes().set_aspect('equal')
        plt.savefig( pic_dir+plt_file, dpi=300, transparent=True, bbox_inches='tight' )
        plt.close()
        
# %% Plot WABBIT files:

files = glob.glob(wdir+'test/'+'/*.h5')
files.sort()
Nt=len(files)
Npics=2
step=Nt//Npics
for iter,file in enumerate(files[::step]):
    plt_file = file.split("/")[-1]
    plt_file = plt_file.replace('h5','png')
    fig, ax = plt.subplots() 
    ax,cb = wt.plot_wabbit_file(file,cmap=fc,caxis_symmetric=True,dpi=300,shading='gouraud',caxis=[-10,10],block_edge_alpha=0.2,colorbar_orientation="horizontal",title=False)
    ax.set_title("Snapshot $u^\epsilon(\mathbf{x},t_i)$ at $i="+str(iter*step)+"$")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_aspect('equal')
    cb.set_label("vorticity [1/s]")
    plt.savefig( pic_dir+plt_file, dpi=300, transparent=True, bbox_inches='tight' )
    plt.close()
    
    
# %% Plot Flusi Modes:
files = glob.glob(resdir_flusi_modes+'/*.h5')
files.sort()

fig = plt.figure()
Nxcut = [50, 206]
Nycut = [128, 128]
Nt = len(files)
Npics = Nt
iter=0
step=Nt//Npics
for iter,file in enumerate(files[::step]):
       # flusi.crop_flusi_HDF5(file,Nxcut,Nycut)
        plt_file = file.split("/")[-1]
        plt_file = plt_file.replace('h5','png')
        time, box, origin, data_flusi = insect_tools.read_flusi_HDF5( file )
        data = np.squeeze(data_flusi).T
       
        y = np.linspace(0,box[-1],np.size(data,0))
        x = np.linspace(0,box[-2],np.size(data,1))
        X,Y = np.meshgrid(x,y)
        plt.pcolormesh(X,Y,data, cmap=fc)#'RdBu')
        plt.title("Mode $\phi_{"+str(iter*step+1)+"}(\mathbf{x})$ ")
        plt.clim(-10,10)
        cl=plt.colorbar(orientation="horizontal")
        cl.set_label("vorticity [1/s]")
        plt.xlabel("$x$")
        plt.ylabel("$y$")
        plt.axes().set_aspect('equal')
        plt.savefig( pic_dir+plt_file, dpi=300, transparent=True, bbox_inches='tight' )
        plt.close() 
        
# %% Plot WABBIT MODES:

files = glob.glob(resdir_wPOD_modes+'/'+'/*.h5')
files.sort()
Nt=len(files)
Npics=Nt
step=Nt//Npics
for iter,file in enumerate(files[::step]):
    plt_file = file.split("/")[-1]
    plt_file = plt_file.replace('h5','png')
    fig, ax = plt.subplots() 
    ax,cb = wt.plot_wabbit_file(file,cmap=fc,caxis_symmetric=True,dpi=300, \
                                shading='gouraud',caxis=[-10,10],block_edge_alpha=0.2,\
                                colorbar_orientation="horizontal",title=False)
    ax.set_title("Mode $\phi^\epsilon_{"+str(iter*step)+"}(\mathbf{x})$ ")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_aspect('equal')
    cb.set_label("vorticity [1/s]")
    plt.savefig( pic_dir+plt_file, dpi=300, transparent=True, bbox_inches='tight' )
    plt.close()
    
# %% PLOT WABBIT PROCS:
wt.plot_wabbit_dir(resdir_wPOD_modes,savepng=True,dpi=300,shading='gouraud', \
                   caxis=[0,4],block_edge_alpha=1,colorbar_orientation="horizontal", \
                   gridonly=True,cmap=plt.get_cmap("Accent"))