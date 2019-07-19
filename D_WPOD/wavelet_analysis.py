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
config_id = "069300"
quantity="vorx"
eps_list = [100, 10, 1, 0.1, 0.01, 0.001]
eps_dir_list = [ "eps"+str(eps) for eps in eps_list]
# %% compute error in 2 norm
fname_flusi = glob.glob(resdir_flusi+'/'+quantity+'_*'+config_id+'.h5')
l2error = np.zeros_like(eps_list)
linferror = np.zeros_like(eps_list)
Nblocks = np.zeros_like(eps_list)

for i, eps_dir in enumerate(eps_dir_list):
        
    fname_dense = glob.glob(eps_dir+'/'+quantity+'-dense_*'+config_id+'*.h5')
    fname_sparse = glob.glob(eps_dir+'/'+quantity+'-sparse_*'+config_id+'*.h5')
    
    # compute Lp errors
    l2error[i] = wt.wabbit_error_vs_flusi(fname_dense[0], fname_flusi[0], norm=2, dim=2)   
    linferror[i] = wt.wabbit_error_vs_flusi(fname_dense[0], fname_flusi[0], norm=np.Inf, dim=2)   

    # compute compression
    Nblocks[i]=sum(wt.block_level_distribution_file( fname_sparse[0] ))

# compute number of dense blokcs
Nblocksdense = sum(wt.block_level_distribution_file( fname_sparse[0] ))
#%% plot Lp error
fig, ax = plt.subplots() 
ax.loglog(eps_list,l2error,'-o', label = "$\Vert u(x) - [u(x)]^\epsilon \Vert_2$")
ax.loglog(eps_list,linferror,'-.*', label = "$\Vert u(x) - [u(x)]^\epsilon \Vert_\infty$")
ax.loglog(eps_list,eps_list, 'k--')
ax.legend()
ax.grid(which='both',linestyle=':')
ax.set_xlabel("$\epsilon$")
ax.set_ylabel("relative error")
plt.savefig( pic_dir+'compression_err4th.png', dpi=300, transparent=True, bbox_inches='tight' )# -*- coding: utf-8 -*-

# %%
fig, ax = plt.subplots() 
ax.semilogx(eps_list,Nblocks/Nblocksdense,'-o')
ax.grid(which='both',linestyle=':')
ax.set_xlabel("$\epsilon$")
ax.set_ylabel("Compression Factor")
plt.savefig( pic_dir+'compression_rate.png', dpi=300, transparent=True, bbox_inches='tight' )# -*- coding: utf-8 -*-
