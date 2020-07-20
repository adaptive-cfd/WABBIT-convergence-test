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
import wabbit_tools as wt
#from wPODdirs import *
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
def plot_wPODerror(Jmax_list, Jmax_dir_list,eps_dir_list,eps_list, dirs, n_star=29):
     # n_star ... mode number at which delta err= |wPODerr - PODerr|
    plt.close("all")
    pic_dir = dirs["images"]
    work = dirs["work"]
    eigval_list={}
    fig=[2,2,2]
    ax=[1,1,1]
    markers = ['o', '.', 'x', '+', 'v', '^', '<', '>', 's', 'd']
    files = {}
    data = {}
    
    eps_dir_list=eps_dir_list[::-1]
    eps_list=eps_list[::-1]
    for Jmax, jmax_dir in zip(Jmax_list, Jmax_dir_list):
        fig[0], ax[0] = plt.subplots()
        fig[1], ax[1] = plt.subplots()
        fig[2], ax[2] = plt.subplots()
        delta_err=np.zeros(np.size(eps_dir_list))
        Neps = len(eps_dir_list)
        clist=[]
        mlist=[]
        for i, eps_dir in enumerate(eps_dir_list):
            #if np.mod(i,2) == 0: continue
            files["L2error"] = work+"/"+jmax_dir+eps_dir + "/L2error.txt"
            files["eigs"] = work+"/"+jmax_dir+eps_dir + "/eigenvalues.txt"
            ########################
            # get data from file
            #######################
            for k,file in files.items():
                f = open(file)
                data[k] = [[float(num) for num in line.split()] for line in f]
                data[k] = np.asarray(data[k])
                f.close
            
            eigs = np.flip(data["eigs"][:,1])
            eigs = eigs[eigs>0]
            PODerr = 1-np.cumsum(eigs)/np.sum(eigs)
            wPODerr = data["L2error"]
            PODerr = np.insert(PODerr,0,1)
            wPODerr =np.insert(wPODerr,0,1)
            delta_err[i]=np.abs(PODerr[n_star]- wPODerr[n_star])
            
        
             ########################
            # save in dictionary
            #######################
            key = eps_list[i]
            
            ########################
            # Do the actual plotting
            #######################
            c1=np.asarray([0.2, 0.7, 1]) # blueish color
            c2=np.asarray([0,0,0])        # black
            alpha=i/Neps
            colors =  c1*(1-alpha)+c2*alpha# RGB,alpha farbverlauf
            clist.append(colors)
            nm = 0# np.mod(i,np.size(markers))
            ax[0].semilogy(PODerr, linestyle='-',marker = markers[nm],\
                        markersize=5,label='$\epsilon ='+str(key) +'$',color=colors,\
                        fillstyle='full')    
            ax[1].semilogy(wPODerr, linestyle='--',marker = markers[nm],\
                        markersize=5,label='$\epsilon ='+str(key) +'$',color=colors,\
                        fillstyle='full')
            # annotation behind the line:
            line = ax[1].lines[-1]
            y = line.get_ydata()[-1]
            x = line.get_xdata()[-3]
            if i <3:
                ax[1].annotate("$\epsilon="+str(key)+"$", xy=(x,y), color=line.get_color(), 
                         xycoords = 'data', textcoords="offset points",
                         size=14, va="center")
                yplt=wPODerr[-1]*0.1 # reference point for start of the arrow
                
            
        
        
        ax[1].semilogy(PODerr, 'k--', mfc='none',\
                        linewidth=1,label='$\epsilon =0$') 
            
        ax[1].set_xlim(-1,len(wPODerr)+5)
        ax[1].set_ylim(PODerr[40],2)
        ## plot arrow pointing downwards to eps -> 0
        ax[1].annotate(" $\epsilon\\to0 $",
            xy=(len(wPODerr)+2, PODerr[len(wPODerr)+2]), xycoords='data',
            xytext=(len(wPODerr)+2, yplt), textcoords='data',
            va="center", ha="center",
            arrowprops=dict(arrowstyle="->", color="0",
                                shrinkA=10, shrinkB=5,
                                patchA=None, patchB=None,
                                connectionstyle="arc3,rad=-0",
                                ),
                )
        

        
        ax[0].set_title("$J_\mathrm{max}="+str(Jmax)+"$")
        ax[1].set_title("$J_\mathrm{max}="+str(Jmax)+"$")
        ax[2].set_title("$J_\mathrm{max}="+str(Jmax)+"$")
        
        #ax[0].legend(bbox_to_anchor=(1.04, 1))
       # ax[1].legend(bbox_to_anchor=(1.04, 1))
  
        ax[0].grid(which='both',linestyle=':')
        ax[1].grid(which='both',linestyle=':')
        ax[2].grid(which='both',linestyle=':')
        
        ax[0].set_ylabel("relative truncation error:\n $\mathcal{E}_{\mathrm{POD}}={\sum_{k=r+1}^N \lambda^\epsilon_k}/{\sum_k \lambda^\epsilon_k}$")
        ax[1].set_ylabel("relative energie error:\n $\mathcal{E}_{\mathrm{wPOD}}$")
        ax[0].set_xlabel("Modes $r$")
        ax[1].set_xlabel("Modes $r$")
        
        ##########################
        # plot delta_POD error
        ##########################
        if eps_list[-1]==0:
            ax[2].scatter(eps_list[:-1],delta_err[:-1],marker='o',color=clist[:-1],)
            alpha = wt.logfit(eps_list[:-1],delta_err[:-1])
            ax[2].loglog(eps_list[:-1],10**alpha[1]*np.asarray(eps_list[:-1])**alpha[0],"k--",label="$\mathcal{O}(\epsilon^{"+str(np.round(alpha[0],2)[0])+"})$")
        else:
            ax[2].loglog(eps_list,delta_err,"*",color=clist)
            alpha = wt.logfit(eps_list[:],delta_err[:])
            ax[2].loglog(eps_list,10**alpha[1]*np.asarray(eps_list)**alpha[0],"k--",label="$\mathcal{O}(\epsilon^{"+str(np.round(alpha[0],2)[0])+"})$")

            
        ax[2].set_xlabel("threshold $\epsilon$")
        ax[2].set_ylabel("$\mid \mathcal{E}_{\mathrm{POD}}-\mathcal{E}_{\mathrm{wPOD}} \mid$")
        ax[2].legend()
    #   
        fig[0].savefig(pic_dir+"/PODerror_J"+str(Jmax)+".eps", dpi=300, transparent=True, bbox_inches='tight' )
        fig[1].savefig(pic_dir+"/wPODerror_J"+str(Jmax)+".eps", dpi=300, transparent=True, bbox_inches='tight' )
        fig[2].savefig(pic_dir+"/deltaPODerror_J"+str(Jmax)+".eps", dpi=300, transparent=True, bbox_inches='tight' )
    
    
        fig[0].show()
        fig[1].show()
        fig[2].show()
    
    return delta_err[::-1],clist[::-1]

def plot_errorWavelet_vs_wPOD(eps_list,delta_err,l2error,clist=None,fname=None,Jmax=5):
            
    fig, ax = plt.subplots()
    ax.scatter(eps_list,delta_err,marker='o',color=clist,markersize=5,
               label="$\mid \mathcal{E}_{\mathrm{POD}}(r^*,0)-\mathcal{E}_{\mathrm{wPOD}}(r^*,\epsilon)\mid$")
    ax.loglog(eps_list,l2error**2,'r-x',label="$\mathcal{E}_{\mathrm{wavelet}(\epsilon)}$")
    ax.loglog(eps_list,eps_list**2,'k--',label="$\epsilon^2$")
    ax.set_title("$J_\mathrm{max}="+str(Jmax)+"$")
    ax.set_xlabel("threshold $\epsilon$")
    ax.set_ylabel("error")
    ax.grid(which='both',linestyle=':')
    ax.legend()
    fig.savefig(fname, dpi=300, transparent=True, bbox_inches='tight' )
    return 0
        
        