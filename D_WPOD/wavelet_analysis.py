#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 12:36:33 2019

@author: Philipp Krah

Tasks of this skript:
    * Analysis of compression error and compression rate for different eps
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
###############################################################################
# Specify:
config_id = "059100"
quantity="vorx"

Ne=5 ## offset for l2error 

Jmax_dir_list = [ "Jmax"+str(Jmax) for Jmax in Jmax_list]
eps_dir_list = [ "eps"+str(eps) for eps in eps_list]
# %% compute error in 2 norm
fname_flusi = glob.glob(resdir_flusi+'/'+quantity+'_*'+config_id+'.h5')
NJmax = len(Jmax_list)
Neps = len(eps_list)
l2error = np.zeros([Neps])

linferror = np.zeros([Neps])
Nblocks = np.zeros([Neps])

fig1, ax1 = plt.subplots() 
fig2, ax2 = plt.subplots() 

l2plt=[0]*len(Jmax_list)
linfplt=[0]*len(Jmax_list)
for j, jmax_dir in enumerate(Jmax_dir_list):
    for i, eps_dir in enumerate(eps_dir_list):
        print(i)
        fname_dense = glob.glob(jmax_dir+"/"+eps_dir+'/'+quantity+'-dense_*'+config_id+'*.h5')
        fname_sparse = glob.glob(jmax_dir+"/"+eps_dir+'/'+quantity+'-sparse_*'+config_id+'*.h5')
        fname_ref = glob.glob(jmax_dir+"/"+eps_dir+'/'+quantity+'_*'+config_id+'*.h5')

        # compute Lp errors
    
        l2error[i] = wt.wabbit_error_vs_flusi(fname_dense[0],fname_flusi[0], norm=2, dim=2)  
        #  l2error[i] = wt.wabbit_error_vs_(fname_ref[0],fname_dense[0], norm=2, dim=2)   
        linferror[i] = wt.wabbit_error_vs_flusi(fname_dense[0],fname_flusi[0], norm=np.inf, dim=2)   

        # compute compression
        Nblocks[i]=sum(wt.block_level_distribution_file( fname_sparse[0] ))

    # compute number of dense blokcs
    Nblocksdense = sum(wt.block_level_distribution_file( fname_dense[0] ))
    
    ##############################################################################
    #    Plot
    ##############################################################################
    #### plot Lp error
    l2plt[j], = ax1.loglog(eps_list[Ne:],l2error[Ne:],'-o', label ="$J_\mathrm{max}="+str(Jmax_list[j])+'$')
    linfplt[j], = ax1.loglog(eps_list[Ne:],linferror[Ne:],'-.*', label = "$J_\mathrm{max}="+str(Jmax_list[j])+'$')

    ####  plot compression rate
    ax2.semilogx(eps_list,Nblocks/Nblocksdense,'-o', label = "$J_\mathrm{max}="+str(Jmax_list[j])+'$')


# Create a legend for the first line.
l2_legend = ax1.legend(handles=l2plt, loc='lower right',title="$\Vert u(x) - [u(x)]^\epsilon \Vert_2$",fontsize='small')
ax1.add_artist(l2_legend)
linf_legend = ax1.legend(handles=linfplt, loc='upper left',title="$\Vert u(x) - [u(x)]^\epsilon \Vert_\infty$",fontsize='small')
ax1.loglog(eps_list,eps_list, 'k--')
ax1.grid(which='both',linestyle=':')
ax1.set_xlabel("$\epsilon$")
ax1.set_ylabel("relative error")

ax2.legend()
ax2.grid(which='both',linestyle=':')
ax2.set_xlabel("$\epsilon$")
ax2.set_ylabel("Compression Factor")
ax1.set_xlim=ax2.get_xlim()


################################################
# save figure
###############################################
fig1.savefig( pic_dir+'compression_err4th.png', dpi=300, transparent=True, bbox_inches='tight' )# -*- coding: utf-8 -*-

fig2.savefig( pic_dir+'compression_rate.png', dpi=300, transparent=True, bbox_inches='tight' )# -*- coding: utf-8 -*-
