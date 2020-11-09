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
from wPOD.sci_notation import sci_notation
###############################################################################
# LATEX FONT
font = {'family' : 'serif',
        'size'   : 24}
rc('font',**font)
rc('text', usetex=True)   
###############################################################################
###############################################################################

# %% get the data
def plot_wPODerror(Jmax_list, Jmax_dir_list,eps_dir_list,eps_list, dirs, n_star=25, \
                   eps_list_plot=[],show_legend=False, show_title=False, \
                       alternate_markers=False,ref_eigvals=False, eigs_larger_eps=False):
     # n_star ... mode number at which delta err= |wPODerr - PODerr|
    plt.close("all")
    pic_dir = dirs["images"]
    work = dirs["work"]
    eigval_list={}
    fig=[2,2,2]
    ax=[1,1,1]
    markers = ['.', 'x', '+', 'v', '^', '<','*', '>', 's', 'd']
    files = {}
    data = {}

    eps_dir_list=eps_dir_list[::-1]
    eps_list=eps_list[::-1]
    Neps = len(eps_list)
    for Jmax, jmax_dir in zip(Jmax_list, Jmax_dir_list):
        fig[0], ax[0] = plt.subplots()
        fig[1], ax[1] = plt.subplots()
        fig[2], ax[2] = plt.subplots()
        delta_err={"mean": np.zeros(Neps),"r*": np.zeros(Neps), "eps": np.zeros(Neps),
                   "errorbar_asymmetric" : [np.zeros(Neps-1),np.zeros(Neps-1)]}
                   
        Neps = len(eps_dir_list)
        clist=[]
        mlist=[]
        if ref_eigvals is False:
            file_ref = work+"/"+jmax_dir+eps_dir_list[-1] + "/L2error.txt"
            print("reference file is: ", file_ref)
            f = open(file_ref)
            data_ref = [[float(num) for num in line.split()] for line in f]
            data_ref = np.asarray(data_ref)
            PODerr_ref=np.insert(data_ref,0,1)
        else:
            file_ref = work+"/"+jmax_dir+eps_dir_list[-1] + "/eigenvalues.txt"
            print("reference file is: ", file_ref)
            f = open(file_ref)
            data_ref = [[float(num) for num in line.split()] for line in f]
            data_ref = np.asarray(data_ref)
            eigs_ref = np.flip(data_ref[:,1])
            eigs_ref = eigs_ref[eigs_ref>0]
            PODerr_ref = 1-np.cumsum(eigs_ref)/np.sum(eigs_ref)
            PODerr_ref = np.insert(PODerr_ref,0,1)

        #eigs_ref = np.flip(data_ref[:,1])
        #eigs_ref = eigs_ref[eigs_ref>0]
     #   PODerr_ref = 1-np.cumsum(eigs_ref)/np.sum(eigs_ref)
        for i, eps in enumerate(eps_list[::-1]):
            #if np.mod(i,2) == 0: continue
            eps_dir = "/eps%1.1e"%eps 
            files["L2error"] = work+"/"+jmax_dir+eps_dir+ "/L2error.txt"
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
            
            Nranks=n_star
            PODerr = 1-np.cumsum(eigs)/np.sum(eigs)
            wPODerr = data["L2error"]
            PODerr = np.insert(PODerr,0,1)
            wPODerr =np.insert(wPODerr,0,1)
            if( eigs_larger_eps):
                Nranks=np.where(PODerr<eps/0.1)[0][0]
                if Nranks > len(wPODerr):
                    Nranks = len(wPODerr)
                if Nranks ==1:
                    Nranks+=1
                    
            print(eps_dir,eps,Nranks,PODerr[Nranks])
            delerr = np.abs(PODerr_ref[1:Nranks]- wPODerr[1:Nranks])
            mean_err = np.mean(delerr)
            delta_err["mean"][i]= mean_err
            delta_err["eps"][i]= eps
            delta_err["r*"][i]=np.abs(PODerr_ref[Nranks-1]- wPODerr[Nranks-1])
            if i < Neps-1:
                delta_err["errorbar_asymmetric"][1][i]=np.max(delerr) - mean_err
                delta_err["errorbar_asymmetric"][0][i]=mean_err - np.min(delerr) 
            
            ########################
            # Do the actual plotting
            #######################
            c1=np.asarray([0.2, 0.7, 1]) # blueish color
            c2=np.asarray([0,0,0])        # black
            alpha=i/Neps
            colors =  c2*(1-alpha)+c1*alpha# RGB,alpha farbverlauf
            clist.append(colors)
            if alternate_markers:
                nm = np.mod(i,np.size(markers))
            else:
                nm = 0
            mlist.append(markers[nm])
            if eps in eps_list_plot:
                ax[0].semilogy(PODerr, linestyle='',marker = markers[nm],\
                        markersize=9,label='$\epsilon = $'+sci_notation(eps) +'',color=colors,\
                        fillstyle='full')    
                ax[1].semilogy(wPODerr[:Nranks], linestyle='',marker = markers[nm],\
                        markersize=9,label='$\epsilon = $'+sci_notation(eps) +'',color=colors,\
                        fillstyle='full')
                line = ax[1].lines[-1]
            else:
                pass
                #ax[0].semilogy(PODerr, linestyle='',marker = markers[nm],\
                #        markersize=5,color=colors,\
                #        fillstyle='full')    
                # ax[1].semilogy(wPODerr[:Nranks], linestyle='',marker = markers[nm],\
                #         markersize=5,color=colors,\
                #         fillstyle='full')
                #line = ax[1].lines[-1]
            # annotation behind the line:
            
            if eps in eps_list_plot and not show_legend:
                x = line.get_xdata()[-3]
                y = line.get_ydata()[-1]
                ax[1].annotate("$\\epsilon= $"+sci_notation(eps)+" ", xy=(x,y), #color=line.get_color(), 
                         xycoords = 'data', textcoords="offset points",
                         size=14, va="center")
            yplt=wPODerr[-1]*0.1 # reference point for start of the arrow
                
            
        
        
        ax[1].semilogy(PODerr_ref, 'k--', mfc='none',\
                        linewidth=1,label='$\mathcal{E}_{\mathrm{POD}},\epsilon =0.0$') 
        ax[0].semilogy(PODerr_ref, 'k--', mfc='none',\
                        linewidth=1,label='$\mathcal{E}_{\mathrm{POD}},\epsilon =0.0$') 
        
            
        ax[1].set_xlim(-1,len(wPODerr)+10)
        ax[1].set_ylim(PODerr[len(wPODerr)+1],2)
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
        

        if show_title:
            ax[0].set_title("$J_\mathrm{max}="+str(Jmax)+"$")
            ax[1].set_title("$J_\mathrm{max}="+str(Jmax)+"$")
            ax[2].set_title("$J_\mathrm{max}="+str(Jmax)+"$")
        
        if ref_eigvals is True:
            ax[0].set_ylim([PODerr_ref[-2]/10, ax[0].get_ylim()[1]])
        #ax[0].legend(bbox_to_anchor=(1.04, 1))
       # ax[1].legend(bbox_to_anchor=(1.04, 1))
  
        ax[0].grid(which='both',linestyle=':')
        ax[1].grid(which='both',linestyle=':')
        ax[2].grid(which='both',linestyle=':')
        
        ax[0].set_ylabel("relative truncation error:\n $\mathcal{E}_{\mathrm{POD}}$")
        ax[1].set_ylabel("relative energie error:\n $\mathcal{E}_{\mathrm{wPOD}}$")
        ax[0].set_xlabel("modes $r$")
        ax[1].set_xlabel("modes $r$")
        ##########################
        # plot delta_POD error
        ##########################
        if eps_list[-1]==0:
          
            alpha = [1,1]#wt.logfit(eps_list[:-1],delta_err["r*"][:-1])
            #ax[2].loglog(eps_list[:-1],10**alpha[1]*np.asarray(eps_list[:-1])**alpha[0],"k--",label="$\mathcal{M}_{r^*}\\times\epsilon^{"+str(np.round(alpha[0],1)[0])+"}$ with $\mathcal{M}_{r^*}=10^{"+str(int(alpha[1]))+"}$,\n $r^*=30$")
            #ax[2].loglog(eps_list[:-1],np.asarray(eps_list[:-1])**2,"k--", label="$\epsilon^2$")
            ax[2].loglog(delta_err["eps"][:-1],np.asarray(delta_err["eps"][:-1]),"k--", label="$\epsilon$")
            ax[2].scatter(delta_err["eps"][:-1],delta_err["r*"][:-1],color=clist[:-1],\
                          label="$\mid \mathcal{E}_{\mathrm{POD}}(r,0)-\mathcal{E}_{\mathrm{wPOD}}(r,\epsilon) \mid$",\
                          marker="x")
            #ax[2].errorbar(delta_err["eps"][:-1],delta_err["mean"][:-1],yerr=delta_err["errorbar_asymmetric"], linestyle="None",color=clist[:-1])            #alpha = wt.logfit(eps_list[:-1],delta_err[:-1])
        else:
            ax[2].scatter(eps_list[:-1],delta_err["mean"][:-1],marker="x",color=clist[:-1],label=r"$\mid \mathcal{E}_{\mathrm{POD}}(r^*,0)-\mathcal{E}_{\mathrm{wPOD}}(r^*,\epsilon) \mid$, $r^*=30$")#,color=clist[:-1])
            ax[2].errorbar(eps_list[:-1],delta_err["mean"][:-1],yerr=delta_err["errorbar_asymmetric"], linestyle="None",color=clist[:-1])            #alpha = wt.logfit(eps_list[:-1],delta_err[:-1])
            #alpha = wt.logfit(eps_list[:],delta_err[:])
            #ax[2].loglog(eps_list,10**alpha[1]*np.asarray(eps_list)**alpha[0],"k--",label="$\mathcal{O}(\epsilon^{"+str(np.round(alpha[0],2)[0])+"})$")
            ax[2].loglog(eps_list[:-1],np.asarray(eps_list[:-1])**2,"k--", label="$\epsilon^2$")
            
        ax[2].set_xlabel(r"threshold $\epsilon$")
        ax[2].set_ylabel(r"error")
        ax[2].legend(loc="lower right",fontsize=18)
        
        if show_legend:
            #pass
            ax[1].legend(loc="upper right",fontsize=16, ncol=1)
            #ax[0].legend(loc="lower left",fontsize="17")
    #   
        if not os.path.exists(pic_dir):
            os.mkdir(pic_dir)
            
    
        fig[0].show()
        fig[1].show()
        fig[2].show()
    
        fig[0].savefig(pic_dir+"/PODerror_J"+str(Jmax)+".svg", dpi=300, transparent=True, bbox_inches='tight' )
        fig[1].savefig(pic_dir+"/wPODerror_J"+str(Jmax)+".svg", dpi=300, transparent=True, bbox_inches='tight' )
        fig[2].savefig(pic_dir+"/deltaPODerror_J"+str(Jmax)+".svg", dpi=300, transparent=True, bbox_inches='tight' )
    
    return delta_err,clist[::-1]

def plot_errorWavelet_vs_wPOD(eps_list,delta_err,l2error,clist=None,fname=None,Jmax=5):
            
    fig, ax = plt.subplots()
    ax.scatter(eps_list,delta_err,marker='o',color=clist,markersize=5,
               label="$\mid \mathcal{E}_{\mathrm{POD}}(r^*,0)-\mathcal{E}_{\mathrm{wPOD}}(r^*,\epsilon)\mid$")
    ax.loglog(eps_list,l2error**2,'r-x',label="$\mathcal{E}_{\mathrm{wavelet}(\epsilon)}$")
    ax.loglog(eps_list,eps_list**2,'k--',label="$\epsilon^2$")
    ax.set_title(r"$J_\mathrm{max}="+str(Jmax)+"$")
    ax.set_xlabel(r"threshold $\epsilon$")
    ax.set_ylabel(r"error")
    ax.grid(which='both',linestyle=':')
    ax.legend()
    fig.savefig(fname, dpi=300, transparent=True, bbox_inches='tight' )
    return 0
        
def read_wPOD_error(params,dirs,ref_eigvals=False):
    from os import path
    
    data = {}
    N_eps  = len(params.eps_list)
    N_Jmax = len(params.jmax_list)
    PODerr_dict= {}
    wPODerr_dict = {}
    delta_PODerr_dict = {}
    files = {}
    
    for n, Jmax in enumerate(params.jmax_list):
        if params.eps_list[0] == 0:
            if ref_eigvals is False:
                file_ref = dirs["work"]+"Jmax%d"%Jmax+"/eps%1.1e"%params.eps_list[0] + "/L2error.txt"
                print("reference file is: ", file_ref)
                f = open(file_ref)
                data_ref = [[float(num) for num in line.split()] for line in f]
                data_ref = np.asarray(data_ref)
                PODerr_ref=np.insert(data_ref,0,1)
            else:
                file_ref = dirs["work"]+"Jmax%d"%Jmax+"/eps%1.1e"%params.eps_list[0] + "/eigenvalues.txt"
                print("reference file is: ", file_ref)
                f = open(file_ref)
                data_ref = [[float(num) for num in line.split()] for line in f]
                data_ref = np.asarray(data_ref)
                eigs_ref = np.flip(data_ref[:,1])
                eigs_ref = eigs_ref[eigs_ref>0]
                PODerr_ref = 1-np.cumsum(eigs_ref)/np.sum(eigs_ref)
                PODerr_ref = np.insert(PODerr_ref,0,1)
            
            

        else:
            print("reference file not found")
            return 0
    
        for m, eps in enumerate(params.eps_list):
            files["L2error"] = dirs["work"]+"Jmax%d"%Jmax+"/eps%1.1e"%eps + "/L2error.txt"
            files["eigs"] = dirs["work"]+"Jmax%d"%Jmax+"/eps%1.1e"%eps + "/eigenvalues.txt"
            ########################
            # get data from file
            #######################
            for k,file in files.items():
                if path.isfile(file):
                    f = open(file)
                    data[k] = [[float(num) for num in line.split()] for line in f]
                    data[k] = np.asarray(data[k])
                    f.close
                else:
                    data[k] = np.NaN
            
            if not np.any(np.isnan(data["eigs"])):    
                eigs = np.flip(data["eigs"][:,1])
                eigs = eigs[eigs>0]
                PODerr = 1-np.cumsum(eigs)/np.sum(eigs)
                PODerr_dict[Jmax,eps] = np.insert(PODerr,0,1)
            if not np.any(np.isnan(data["L2error"])):    
                wPODerr = data["L2error"]
                wPODerr_dict[Jmax,eps] = np.insert(wPODerr,0,1)
                
            delta_PODerr_dict[Jmax,eps] = abs(PODerr_ref[:len(wPODerr_dict[Jmax,eps])]-wPODerr_dict[Jmax,eps])
            
    return wPODerr_dict,PODerr_dict,delta_PODerr_dict