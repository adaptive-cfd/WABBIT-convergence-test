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
from wPODdirs import *
from matplotlib import rc

font = {'family' : 'sans-serif',
        'sans-serif':['Helvetica'],
        'size'   : 22}
rc('font',**font)
rc('text', usetex=True)     


pic_dir = "./images/"
eps_dir_list = [ "eps"+str(eps) for eps in eps_list]
#wdir = "/home/krah/develop/WABBIT/"
wdir = "~/savart/develop/WABBIT/"
resdir_flusi=home+"/develop/results/cyl/wPOD/vor_up/"
resdir_flusi_modes="../results/cyl/vorticity_POD2/modes/"
resdir_wPOD_modes=eps_dir_list[2]
resdir_wabbit =resdir_flusi + "_adapt/"
daedalus_pic_dir = "../../results/daedalus_logo/"

fc = farge_colormaps.farge_colormap_multi( taille=600,limite_faible_fort=0.2, etalement_du_zero=0.04, type='vorticity' )





# %% Plot flusi original files

files = glob.glob(resdir_flusi+'/*.h5')
files.sort()

fig = plt.figure()
Nxcut = [50, 206]
Nycut = [128, 128]
Nupsample = [1024*3,1024]
Nt = len(files)
Npics = 2
iter=0
step=1#Nt//Npics
for iter,file in enumerate(files[::step]):
        #flusi.crop_flusi_HDF5(file,Nxcut,Nycut)
        flusi.resample_flusi_HDF5(file,Nupsample)
#        plt_file = file.split("/")[-1]
#        plt_file = plt_file.replace('h5','png')
#        time, box, origin, data_flusi = insect_tools.read_flusi_HDF5( file )
#        data = np.squeeze(data_flusi).T
#        y = np.linspace(0,box[-1],np.size(data,0))
#        x = np.linspace(0,box[-2],np.size(data,1))
#        X,Y = np.meshgrid(x,y)
#        plt.pcolormesh(X,Y,data, cmap=fc)#'RdBu')
#        plt.title("Snapshot $u(\mathbf{x},t_i)$ at $i="+str(iter*step)+"$")
#        plt.clim(-10,10)
#        cl=plt.colorbar(orientation="horizontal")
#        cl.set_label("vorticity [1/s]")
#        plt.xlabel("$x$")
#        plt.ylabel("$y$")
#        plt.axes().set_aspect('equal')
#        plt.show()
#       # plt.pause(0.5)
#        plt.savefig( pic_dir+plt_file, dpi=300, transparent=True, bbox_inches='tight' )
#        plt.close()
#        
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

files =  []
save_dir = "mode/"
step=2
if not os.path.exists(pic_dir+save_dir):
    os.mkdir(pic_dir+save_dir) # make directory for saving the files

for jmax in Jmax_list:
    for index,val in enumerate(eps_dir_list):
        files = glob.glob('Jmax'+str(jmax)+'/'+eps_dir_list[index]+'/'+'/*.h5')
        files.sort()
        Nt=len(files)
        Npics=Nt
        for iter,file in enumerate(files[::step]):
            plt_file = file.split("/")[-1]
            plt_file = plt_file.replace("mode1","mode""-jmax"+str(jmax)+"-"+eps_dir_list[index])
            plt_file = plt_file.replace('.h5','.png')
            plt_file_procs = plt_file.replace("mode","mode-procs")
           
            ####################
            # plot field
            ####################
            fig, ax = plt.subplots() 
            ax,cb = wt.plot_wabbit_file(file,cmap=fc,caxis_symmetric=True,dpi=300, ticks=False,\
                                shading='gouraud',caxis=[-10,10],block_edge_alpha=0.2,\
                                colorbar_orientation="horizontal",colorbar=False,title=False)
            ax.set_aspect('equal')
            plt.savefig( pic_dir+"/"+save_dir+"/"+plt_file, dpi=300, transparent=True, bbox_inches='tight' )
            plt.close()
            
            ###################
            # plot procs
            ###################
            fig, ax = plt.subplots() 
            ax,cb = wt.plot_wabbit_file(file,dpi=300,shading='gouraud', \
                   caxis=[0,4],block_edge_alpha=1,colorbar_orientation="horizontal", \
                   gridonly=True,cmap=plt.get_cmap("Accent"),ticks=False,\
                   colorbar=False,title=False)
            ax.set_aspect('equal')
            plt.savefig( pic_dir+save_dir+plt_file_procs, dpi=300, transparent=True, bbox_inches='tight' )
            plt.close()
    
    
    
# %% DAEDALUS pics
    
    
files = glob.glob(daedalus_pic_dir+'/'+'/dadd.h5')
files.sort()
Nt=len(files)
Npics=Nt
step=Nt//Npics
for iter,file in enumerate(files[::step]):
    plt_file = file.split("/")[-1]
    plt_file_procs = plt_file.replace('.h5','2procs.png')
    plt_file_field = plt_file.replace('.h5','2field.png')
    fig, ax = plt.subplots() 
    ax,cb = wt.plot_wabbit_file(file,dpi=300,shading='gouraud', \
                   caxis=[0,4],block_edge_alpha=1,colorbar_orientation="horizontal", \
                   gridonly=True,cmap=plt.get_cmap("Accent"),ticks=False,\
                   colorbar=False,title=False)
    #ax.set_title("Mode $\phi^\epsilon_{"+str(iter*step)+"}(\mathbf{x})$ ")
   # ax.set_xlabel("$x$")
    #ax.set_ylabel("$y$")
    ax.set_aspect('equal')
   # cb.set_label("vorticity [1/s]")
    plt.savefig( pic_dir+plt_file_procs, dpi=300, transparent=True, bbox_inches='tight' )
    plt.close()
    
    ax,cb = wt.plot_wabbit_file(file,dpi=300,shading='gouraud',\
                  block_edge_alpha=1,colorbar_orientation="horizontal", \
                   gridonly=False,cmap=plt.get_cmap("Blues_r"),ticks=False,\
                   colorbar=False,title=False,caxis=[0,1])
    #ax.set_title("Mode $\phi^\epsilon_{"+str(iter*step)+"}(\mathbf{x})$ ")
   # ax.set_xlabel("$x$")
    #ax.set_ylabel("$y$")
    ax.set_aspect('equal')
   # cb.set_label("vorticity [1/s]")
    plt.savefig( pic_dir+plt_file_field, dpi=300, transparent=True, bbox_inches='tight' )
    plt.close()