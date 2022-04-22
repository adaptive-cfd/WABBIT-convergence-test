#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 10:55:09 2019

@author: krah
"""
import numpy as np
import wabbit_tools as wt
import matplotlib.pyplot as plt
import re
import os
import glob


###############################################################################
# %% Change parameters here
###############################################################################
# directories needed
dirs= {
       'wabbit' : "~/develop/WABBIT/" ,         # directory of wabbit
       'work'   : "./",                         # where to save big data files
       'images' : "./"                           #pictures are saved here
       }

# setup for wabbit call
wabbit_setup = {
        'mpicommand' : "mpirun -np 8",
        'memory'     : "--memory=8GB"
        }
# parameters to adjust
class params:
    
   eps_list  = np.logspace(-4,0,10)    # threshold of adaptation
   jmax_list = [4,5,6]        # maximal tree level
    
   class domain:
        N = [2048, 2048]            # number of points for 3d use 3 elements
        L = [1.0, 1.0]              # length of domain
  
   #exp(-(x²+y²)/2/sigma²)
   def init(x,L):
       sigma = np.asarray(L)*0.01
       x0    = [Li/2 for Li in L]
       xrel  = [x-x0 for x,x0 in zip(x,x0)]
       field = np.ones_like(x[0])
       for x,s in zip(xrel,sigma):
           field *= np.exp(-np.power(x,2)/(2*s**2))
       return field
    
    ## sin(1/x)*sin(1/y)
   def init2(x,L):                             # this function will be initialized on the domain
        sigma = np.asarray(L)*0.01
        x0    = [Li/2 for Li in L]
        xrel  = [x-x0 for x,x0 in zip(x,x0)]
        field = np.ones_like(x[0])
        for x,s in zip(xrel,sigma):
            field *= np.sin(-np.divide(1,np.abs(x)))
        return field     
    
      ## (x**2+y**2)*heaviside(x)
   def init3(x,L):                             # this function will be initialized on the domain
        sigma = np.asarray(L)*0.01
        x0    = [Li/2 for Li in L]
        xrel  = [x-x0 for x,x0 in zip(x,x0)]
        field = (xrel[0]**2+xrel[1]**2)*np.heaviside(xrel[0],0)
        return field     
###############################################################################
# %% 
###############################################################################





def wabbit_adapt(dirs, params, wabbit_setup):
    
    
    mpicommand = wabbit_setup["mpicommand"]
    memory     = wabbit_setup["memory"]
    wdir = dirs["wabbit"]
    work = dirs["work"]
    jmax_list = params.jmax_list
    # set initial gird
    dim = len(params.domain.L)
    x = [np.linspace(0,params.domain.L[d],params.domain.N[d]) \
         for d in range(dim) ]
    X = np.meshgrid(*x)
    phi = params.init(X,params.domain.L)
    Bs = wt.field_shape_to_bs(params.domain.N,jmax_list[-1])
    
    # create reference file
    file_ref = wt.dense_to_wabbit_hdf5(phi,work+"/phi", Bs, params.domain.L,0)
    
    
    params.Bs_list = []
    l2error=np.zeros([len(params.jmax_list),len(params.eps_list)])
    linferror=np.zeros_like(l2error)
    compress = np.zeros_like(l2error)
    for j,jmax in enumerate(params.jmax_list):
        Bs = wt.field_shape_to_bs(params.domain.N,jmax)
        params.Bs_list.append(Bs)        
        # adapt field for different eps using wabbit-post -dense-to-sparse
        print("\n\n###################################################################")
        print("###################################################################")
        print( "\t\tJmax: ", jmax, "\n\n")
        for k,eps in enumerate(params.eps_list):
            # create dense field and save to work
            fname = "phi-j"+str(jmax)+"-eps"
            file = work +"/"+fname
            file = wt.dense_to_wabbit_hdf5(phi,file, Bs, params.domain.L,eps*100)
            command = mpicommand + " " +  wdir + \
                 "wabbit-post --dense-to-sparse --eps=" + str(eps) + " --order=CDF44 --eps-norm=Linfty " + memory + \
                  " --files="+ file + ">adapt.log"
                  # command for densing file again
            dense_command = mpicommand + " " +  wdir + \
                 "wabbit-post --sparse-to-dense " + \
                  " "+ file + " "+file + ">>adapt.log"
            # -----------------------------
            # Execute Command
            # -----------------------------
            print("\n\n###################################################################")
            print("\t\teps =",eps)
            print("###################################################################")
            print("\n",command,"\n\n")
            success = os.system(command)   # execute command
            if np.mod(k,4)==0:
                 wt.plot_wabbit_file(file,savepng=True)
            compress[j,k] = sum(wt.block_level_distribution_file( file ))
             
            print("\n",dense_command,"\n\n")
            success += os.system(dense_command) # dense the file for comparison to original
            if success != 0:
                print("command did not execute successfully")
                return
            
            # ---------------------------
            # compare to original file
            # --------------------------
            l2error[j,k] = wt.wabbit_error_vs_wabbit(file_ref,file, norm=2, dim=2)  
            linferror[j,k] = wt.wabbit_error_vs_wabbit(file_ref,file, norm=np.inf, dim=2)  
            compress[j,k] /=  sum(wt.block_level_distribution_file( file ))
        
            # delete file
            os.system("rm "+ file)
    
    # delete reference file
    os.system("rm " + file_ref)
    return l2error,linferror,compress


if __name__ == "__main__":

    l2error, linferror, compress = wabbit_adapt(dirs, params, wabbit_setup)        
    
    
    # %%
    plt.close("all")
    fig1, ax1 = plt.subplots() 
    fig2, ax2 = plt.subplots() 
    
    l2plt=[0]*len(params.jmax_list)
    linfplt=[0]*len(params.jmax_list)
    for j, jmax in enumerate(params.jmax_list):
        l2plt[j], = ax1.loglog(params.eps_list,l2error[j,:],'-o', label ="$J_\mathrm{max}="+str(jmax)+'$')
        linfplt[j], = ax1.loglog(params.eps_list,linferror[j,:],'-.*', label = "$J_\mathrm{max}="+str(jmax)+'$')
        ####  plot compression rate
        ax2.semilogx(params.eps_list,compress[j,:],'-o', label = "$J_\mathrm{max}="+str(jmax)+'$')
    
    l2_legend = ax1.legend(handles=l2plt, loc='lower right',title="$\Vert u(x) - [u(x)]^\epsilon \Vert_2$",fontsize='small')
    ax1.add_artist(l2_legend)
    linf_legend = ax1.legend(handles=linfplt, loc='upper left',title="$\Vert u(x) - [u(x)]^\epsilon \Vert_\infty$",fontsize='small')
    ax1.loglog(params.eps_list,params.eps_list, 'k--')
    ax1.grid(which='both',linestyle=':')
    ax1.set_xlabel("$\epsilon$")
    ax1.set_ylabel("relative error")
    
    ax2.legend()
    ax2.grid(which='both',linestyle=':')
    ax2.set_xlabel("$\epsilon$")
    ax2.set_ylabel("Compression Factor")
    #ax1.set_xlim=ax2.get_xlim()


################################################
# save figure
###############################################
#fig1.savefig( pic_dir+'blob_compression_err4th.png', dpi=300, transparent=True, bbox_inches='tight' )# -*- coding: utf-8 -*-

#fig2.savefig( pic_dir+'blob_compression_rate.png', dpi=300, transparent=True, bbox_inches='tight' )# -*- coding: utf-8 -*-