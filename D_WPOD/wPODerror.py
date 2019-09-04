#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 12:43:01 2019

@author: phil

Tasks of this skript:
    * Plot wPODerror
"""
###############################################################################
#           MODULES
###############################################################################
import os
#import farge_colormaps
import glob
from wPODdirs import *
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc
###############################################################################
# LATEX FONT
font = {'family' : 'serif',
        'size'   : 16}
rc('font',**font)
rc('text', usetex=True)   
###############################################################################
###############################################################################

# %% get the data
plt.close("all")
eigval_list={}
fig=[2,2,2]
ax=[1,1,1]
markers = ['o', '.', 'x', '+', 'v', '^', '<', '>', 's', 'd']
files = {}
data = {}
n_star = 29  # mode number at which delta err= |wPODerr - PODerr|
for Jmax, jmax_dir in zip(Jmax_list, Jmax_dir_list):
    fig[0], ax[0] = plt.subplots()
    fig[1], ax[1] = plt.subplots()
    fig[2], ax[2] = plt.subplots()
    delta_err=np.zeros(np.size(eps_dir_list))
    for i, eps_dir in enumerate(eps_dir_list):
        files["L2error"] = jmax_dir+eps_dir + "/L2error.txt"
        files["eigs"] = jmax_dir+eps_dir + "/eigenvalues.txt"
        ########################
        # get data from file
        #######################
        for k,file in files.items():
            f = open(file)
            data[k] = [[float(num) for num in line.split()] for line in f]
            data[k] = np.asarray(data[k])
            f.close
        
        eigs = np.flip(data["eigs"][:,1])
        PODerr = 1-np.cumsum(eigs)/np.sum(eigs)
        wPODerr = data["L2error"]
        delta_err[i]=np.abs(PODerr[n_star]- wPODerr[n_star])
        
    
         ########################
        # save in dictionary
        #######################
        key = eps_list[i]
        
        ########################
        # Do the actual plotting
        #######################
        nm = np.mod(i,np.size(markers))
        ax[0].semilogy(PODerr, linestyle='-',marker = markers[nm], mfc='none',\
                    markersize=5,label='$\epsilon ='+str(key) +'$')    
        ax[1].semilogy(wPODerr, linestyle='--',marker = markers[nm], mfc='none',\
                    markersize=5,label='$\epsilon ='+str(key) +'$')
        
    
    ax[1].semilogy(PODerr, 'k--', mfc='none',\
                    linewidth=1,label='$\epsilon =0$')    
    ax[1].set_xlim(-1,40)
    ax[1].set_ylim(PODerr[40],1)
    
    ax[0].set_title("$J_\mathrm{max}="+str(Jmax)+"$")
    ax[1].set_title("$J_\mathrm{max}="+str(Jmax)+"$")
    ax[2].set_title("$J_\mathrm{max}="+str(Jmax)+"$")
    
    ax[0].legend(bbox_to_anchor=(1.04, 1))
    ax[1].legend(bbox_to_anchor=(1.04, 1))
#   
    ax[0].grid(which='both',linestyle=':')
    ax[1].grid(which='both',linestyle=':')
    
    ax[0].set_ylabel("relative truncation error:\n $\mathcal{E}_{\mathrm{POD}}={\sum_{k=r+1}^N \lambda^\epsilon_k}/{\sum_k \lambda^\epsilon_k}$")
    ax[1].set_ylabel("relative energie error:\n $\mathcal{E}_{\mathrm{wPOD}}$")
    ax[0].set_xlabel("Modes $r$")
    ax[1].set_xlabel("Modes $r$")
    
    ##########################
    # plot delta_POD error
    ##########################
    ax[2].loglog(eps_list,delta_err,"*")
    alpha = wt.logfit(eps_list[:-3],delta_err[:-3])
    ax[2].loglog(eps_list,10**alpha[1]*np.asarray(eps_list)**alpha[0],"k--",label="$\mathcal{O}(\epsilon^{"+str(np.round(alpha[0],2)[0])+"})$")
    ax[2].set_xlabel("threshold $\epsilon$")
    ax[2].set_ylabel("$\mid \mathcal{E}_{\mathrm{POD}}-\mathcal{E}_{\mathrm{wPOD}} \mid$")
    ax[2].legend()
#   
    fig[0].savefig(pic_dir+"PODerror.eps", dpi=300, transparent=True, bbox_inches='tight' )
    fig[1].savefig(pic_dir+"wPODerror.eps", dpi=300, transparent=True, bbox_inches='tight' )
    fig[2].savefig(pic_dir+"deltaPODerror.eps", dpi=300, transparent=True, bbox_inches='tight' )

    fig[0].show()
    fig[1].show()
    fig[2].show()

