#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 12:36:33 2019

@author: Philipp Krah

Tasks of this skript:
    * Plot initial snapshot for different eps
"""
###############################################################################
#           MODULES
###############################################################################
import os
import farge_colormaps
import glob
from wPODdirs import *
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc
import wabbit_tools as wt
###############################################################################
# LATEX FONT
font = {'family' : 'serif',
        'size'   : 18}
rc('font',**font)
rc('text', usetex=True)   
###############################################################################
# COLORMAP:
fc = farge_colormaps.farge_colormap_multi( taille=600,limite_faible_fort=0.2, etalement_du_zero=0.04, type='vorticity' )
###############################################################################
# Specify:
#eps_list = [10, 0.1, 0.01, 0.001, 0.001]
eps_dir_list = [ "eps"+str(eps) for eps in eps_list]
###############################################################################
# %% get the data
quantity="vorx-sparse"
plt.close("all")

for i, eps_dir in enumerate(eps_dir_list[::5]):
    files = glob.glob(eps_dir+'/'+quantity+'_*.h5')
    files.sort()
    for file in files:
        plt_file = file.split("/")[-1]
        plt_file = plt_file.replace(quantity+"_",quantity+"-"+eps_dir_list[i]+"_")
        plt_file = plt_file.replace('h5','png')
        fig, ax = plt.subplots() 
        ax,cb = wt.plot_wabbit_file(file,cmap=fc,caxis_symmetric=True,dpi=300, \
                                    shading='gouraud',caxis=[-10,10], \
                                    block_edge_alpha=0.2,colorbar_orientation="horizontal", \
                                    title=False, ticks=False, colorbar=False)
        ax.set_title("$\epsilon="+str(eps_list[i])+"$")
       # ax.set_xlabel("$x$")
        #ax.set_ylabel("$y$")
        ax.set_aspect('equal')

        #cb.set_label("vorticity [1/s]")
        print("Saving plot to: "+pic_dir+plt_file)
        plt.savefig( pic_dir+plt_file, dpi=300, transparent=True, bbox_inches='tight' )
        plt.show()

##############################################################################