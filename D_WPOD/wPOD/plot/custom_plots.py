#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 09:41:25 2020

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
import matplotlib.gridspec as gs
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
font = {'family' : 'sans-serif',
        'sans-serif':['Helvetica'],
        'size'   : 22}
rc('font',**font)
rc('text', usetex=True)     

fc_default = farge_colormaps.farge_colormap_multi(taille=600,etalement_du_zero=0.04, type='vorticity' )

def plot_modes(files,acoef_file,quantity,quantity_replace,pic_dir="./",fc=fc_default):
    
    f = open(acoef_file)
    data = [[float(num) for num in line.split()] for line in f]
    f.close
    data = np.asarray(data)
    amin = np.amin(data)
    amax = np.amax(data)
    print("min(a_coef):", amin)
    print("max(a_coef):", amax)
    for mode,file in enumerate(files):
        plt_file = file.split("/")[-1]
        plt_file = plt_file.replace(quantity+"_",quantity_replace+"_")
        plt_file = plt_file.replace('h5','png')
    
        fig = plt.figure(44)
        fig.subplots_adjust(wspace=0.05,hspace=0.0)
        gs1 = gs.GridSpec(nrows=6,ncols=2)
        ax1 = plt.subplot(gs1[:5,0])
        ax,cb,hplot = wt.plot_wabbit_file(file,cmap=fc,caxis_symmetric=True,dpi=600, \
                                    shading='gouraud', \
                                    block_edge_alpha=0.1, \
                                    title=False, ticks=False, colorbar=False, fig=fig, ax=ax1)     
        
        
        ax.set_title("$\Psi_{"+str(mode+1)+"}^\epsilon(x,y)$")
       # ax.set_xlabel("$x$")
        #ax.set_ylabel("$y$")y
        ax2 = plt.subplot(gs1[5:,0])
        ########################
        # Do the actual plotting
        #######################
        ax2.plot(data[:,mode], linestyle='-',marker = ".", mfc='none', markersize=5)
        ax2.set_xlabel("$i$")
        ax2.set_ylabel("$a^\epsilon_{"+str(mode+1)+"}(t_i)$")
        ax2.yaxis.tick_right()
        ax2.set_ylim([amin,amax])
        # Add the colorbar outside...

        
        pad, width = 0.02, 0.02
        fig.subplots_adjust(bottom=0.2,top=0.9)
        cax = plt.subplot(gs1[:5,1])
        box = ax1.get_position()
        fig.colorbar(hplot, cax=cax)
        cax.set_position([cax.get_position().xmin, box.ymin, width , box.height])
        cax.axis('tight')
  
               
       
        #cb.set_label("vorticity [1/s]")
        print("Saving plot to: "+pic_dir+plt_file)
        plt.savefig( pic_dir+plt_file, dpi=600, transparent=True, bbox_inches='tight' )
        plt.show()


def plot_vector_modes(files,acoef_file,quantity,quantity_replace,pic_dir="./",fc=fc_default, mode_names=None):
    
    f = open(acoef_file)
    data = [[float(num) for num in line.split()] for line in f]
    f.close
    data = np.asarray(data)
    amin = np.amin(data)
    amax = np.amax(data)
    print("min(a_coef):", amin)
    print("max(a_coef):", amax)
    Ncomponents=len(files)
    Nmodes = len(files[0])
    mode_ax=[]
    for n_mode in range(Nmodes):
        fig = plt.figure(n_mode)
        fig.subplots_adjust(wspace=0.05,hspace=0.05)
        for k in range(Ncomponents):
            file = files[k][n_mode]
            plt_file = file.split("/")[-1]
            plt_file = plt_file.replace(quantity+"_",quantity_replace+"_")
            plt_file = plt_file.replace('h5','png')
        
            gs1 = gs.GridSpec(nrows=Ncomponents+1,ncols=1)
            ax1 = plt.subplot(gs1[k,0])
            ax,cb,hplot = wt.plot_wabbit_file(file,cmap=fc,caxis_symmetric=True,dpi=600, \
                                        shading='flat', block_linewidth=0.25, \
                                        block_edge_alpha=0.4, block_edge_color='k', \
                                        title=False, ticks=False, colorbar=False, fig=fig, ax=ax1)     
            if k==0:
                ax.set_title("$\Psi_{"+str(n_mode+1)+"}^\epsilon(x,y)$")
            if mode_names is not None:
                ax1.set_ylabel(mode_names[k])
           # ax.set_xlabel("$x$")
            #ax.set_ylabel("$y$")y
            mode_ax.append(ax1)
            
            axins1 = inset_axes(ax1,
                    width="2%",  # width = 50% of parent_bbox width
                    height="80%",  # height : 5%
                    loc='right', 
                    bbox_to_anchor=(0.03, 0., 1, 1),
                    bbox_transform=ax1.transAxes,
                    borderpad=0)
            fig.colorbar(hplot, cax=axins1, orientation="vertical")
        # for k,ax1 in enumerate(mode_ax):
        #      pad, width = 0.02, 0.02
        #      fig.subplots_adjust(bottom=0.2,top=0.9)
        #      fig.canvas.draw()        
        #      cax = plt.subplot(gs1[np.mod(k,1),1])
        #      box = ax1.get_position()
        #      cax.set_position([cax.get_position().xmin, box.ymin, 0.01 , box.height])
        #      cax.axis('tight')
        #      fig.colorbar(hplot, cax=cax)
        
          
        ax2 = plt.subplot(gs1[Ncomponents,0])
        ########################
        # Do the actual plotting
        #######################
        ax2.plot(data[:,n_mode], linestyle='-',marker = ".", mfc='none', markersize=5)
        ax2.set_xlabel("$i$")
        ax2.set_ylabel("$a^\epsilon_{"+str(n_mode+1)+"}(t_i)$")
        ax2.yaxis.tick_right()
        ylim=1.2*np.max(np.abs(ax2.get_ylim()))
        ax2.set_ylim([-ylim,ylim])
        # Add the colorbar outside...

        
                   
           
        #cb.set_label("vorticity [1/s]")
        print("Saving plot to: "+pic_dir+plt_file)
        plt.savefig( pic_dir+plt_file, dpi=600, transparent=True, bbox_inches='tight' )
        plt.show()