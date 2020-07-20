#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 10:55:09 2019

@author: krah
"""
import numpy as np
from wPODdirs import *
from wPOD.synthetic_test_case import synthetic_test_case
from wPOD.adaption_test import adaptation_for_different_eps
import farge_colormaps

###############################################################################
# COLORMAP:
fc = farge_colormaps.farge_colormap_multi( taille=600,limite_faible_fort=0.2, etalement_du_zero=0.04, type='vorticity' )
###############################################################################
# directories needed
dirs["work"]+="/bump/"
dirs["images"]+="/bump/"
###############################################################################
# %% Set Parameters
############################################################################### 
class params:
   case = "Bumbs" # choose [Bumbs,Mendez] 
   slow_singular_value_decay=False
   #eps_list  = np.asarray([0]+[float("%1.1e" %eps) for eps in np.logspace(-12,0,6)])  # threshold of adaptation
   eps_list  = np.asarray([float("%1.1e" %eps) for eps in np.logspace(-5,0,10)])  # threshold of adaptation
   jmax_list = [4,5,6]        # maximal tree level
   Nt= 2**7
   
   class domain:
        N = [2**8, 2**8]      # number of points
        L = [30, 30]            # length of domain
  
        
Jmax_dir_list = [ "Jmax"+str(Jmax)+"/" for Jmax in params.jmax_list]
eps_list = params.eps_list
eps_dir_list = [ "eps%1.1e" %eps for eps in eps_list]

file_list = synthetic_test_case(dirs, params, wabbit_setup)
delta_err,clist=plot_wPODerror(params.jmax_list, Jmax_dir_list,eps_dir_list,eps_list, dirs, n_star=1)

h5_fname = file_list[params.Nt//3]
wdir = dirs["wabbit"]
[l2error,sucess]=adaptation_for_different_eps(h5_fname, wdir, eps_list, wabbit_setup['memory'], wabbit_setup['mpicommand'], create_log_file=True)
# %%
# plt.close("all")
# mpicommand = wabbit_setup["mpicommand"]
# memory     = wabbit_setup["memory"]
# wdir = dirs["wabbit"]
# work = dirs["work"]
# jmax_list = params.jmax_list
# # set initial gird
# dim = len(params.domain.L)
# x = [np.linspace(0,params.domain.L[d],params.domain.N[d]) \
#       for d in range(dim) ]
# t = np.linspace(0,5.11,params.Nt)
# [X,Y] = np.meshgrid(*x)

# u, phi, amp = params.init([X,Y],t,params.domain.L, case="Bumbs")
# u_field = np.reshape(u,[*params.domain.N,params.Nt])
# [U,S,V] = np.linalg.svd(u,0)
# l2plt=[0]*len(params.jmax_list)
# linfplt=[0]*len(params.jmax_list)
# for j, jmax in enumerate(params.jmax_list):
#     l2plt[j], = ax1.loglog(params.eps_list,l2error[j,:],'-o', label ="$J_\mathrm{max}="+str(jmax)+'$')
#     linfplt[j], = ax1.loglog(params.eps_list,linferror[j,:],'-.*', label = "$J_\mathrm{max}="+str(jmax)+'$')
#     ####  plot compression rate
#     ax2.semilogx(params.eps_list,compress[j,:],'-o', label = "$J_\mathrm{max}="+str(jmax)+'$')

# l2_legend = ax1.legend(handles=l2plt, loc='lower right',title="$\Vert u(x) - [u(x)]^\epsilon \Vert_2$",fontsize='small')
# ax1.add_artist(l2_legend)
# linf_legend = ax1.legend(handles=linfplt, loc='upper left',title="$\Vert u(x) - [u(x)]^\epsilon \Vert_\infty$",fontsize='small')
# ax1.loglog(params.eps_list,params.eps_list, 'k--')
# ax1.grid(which='both',linestyle=':')
# ax1.set_xlabel("$\epsilon$")
# ax1.set_ylabel("relative error")

# ax2.legend()
# ax2.grid(which='both',linestyle=':')
# ax2.set_xlabel("$\epsilon$")
# ax2.set_ylabel("Compression Factor")


################################################
# save figure
###############################################
#fig1.savefig( pic_dir+'blob_compression_err4th.png', dpi=300, transparent=True, bbox_inches='tight' )# -*- coding: utf-8 -*-

#fig2.savefig( pic_dir+'blob_compression_rate.png', dpi=300, transparent=True, bbox_inches='tight' )# -*- coding: utf-8 -*-
