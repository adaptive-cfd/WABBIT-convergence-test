#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 16:10:16 2020

@author: phil

This skript is for Tommy (:
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# %% adapt one snapshot for different eps: logfile written in adapt .log
def adaptation_for_different_eps(h5_fname, wdir, eps_list, memory, mpicommand, create_log_file=True):
    
    import re
    import os
    import glob
    import wabbit_tools as wt
    # generate new file names 
    print( "Input data file: ", h5_fname, "\n")
    file = h5_fname.split("/")[-1]
    filesparse =file.replace('_','-eps_%1.1e_')
    filedense =file.replace('_','-dense_')
    # command for sparsing file
    command = mpicommand + " " +  wdir + \
             "wabbit-post --dense-to-sparse --eps=%1.1e --order=CDF40 --eps-norm=Linfty " + memory + \
              ' --files="'+ filesparse + '" ' 
    # command for densing file again
    dense_command = mpicommand + " " +  wdir + \
             "wabbit-post --sparse-to-dense " + \
              " "+ filesparse + " "+filedense + ">>adapt.log"
                  
    Neps    = len(eps_list)
    Nblocks = np.zeros([Neps])
    
    l2error = np.zeros([Neps])
    linferror = np.zeros([Neps])
    
    for i,eps in enumerate(eps_list):
        # -----------------------------
        # Prepare to save data from POD
        # -----------------------------
        c       = command % (eps,eps)  # change eps
        dc      = dense_command % eps      # change eps
        fsparse = filesparse%eps
        if create_log_file:
            c += " > adapt.log" 
        
        # copy given file in current directory 
        # os.system("cp "+h5_fname+ " .")
        os.system("cp "+h5_fname+ " ./" + fsparse )
        # -----------------------------
        # Execute Command
        # -----------------------------
        print("\n\n###################################################################")
        print("\t\teps =%1.1e"%eps)
        print("###################################################################")
        print("\n",c,"\n\n")
        success = os.system(c)   # execute command
        
        print("\n",dc,"\n\n")
        success += os.system(dc) # dense the file for comparison to original
        if success != 0:
            print("command did not execute successfully")
            break
        
        
        l2error[i] = wt.wabbit_error_vs_wabbit(file,filedense, norm=2, dim=2)  
        #  l2error[i] = wt.wabbit_error_vs_(fname_ref[0],fname_dense[0], norm=2, dim=2)   
        linferror[i] = wt.wabbit_error_vs_wabbit(file,filedense, norm=np.inf, dim=2)   

        # compute compression
        Nblocks[i]=sum(wt.block_level_distribution_file( fsparse ))

        # compute number of dense blokcs
    Nblocksdense = sum(wt.block_level_distribution_file( file ))
    
    ##############################################################################
    #    Plot
    ##############################################################################
    #### plot Lp error    
    fig1, ax1 = plt.subplots() 
    fig2, ax2 = plt.subplots() 
    Ne = 0
    l2plt, = ax1.loglog(eps_list[Ne:],l2error[Ne:],'-o',label="$\Vert u(x) - [u(x)]^\epsilon \Vert_2$")
    linfplt, = ax1.loglog(eps_list[Ne:],linferror[Ne:],'-.*', label= "$\Vert u(x) - [u(x)]^\epsilon \Vert_\infty$")
    ax1.legend()
    ####  plot compression rate
    ax2.semilogx(eps_list,Nblocks/Nblocksdense,'-o')
    # Create a legend for the first line.
    ax1.loglog(eps_list,eps_list, 'k--')
    ax1.grid(which='both',linestyle=':')
    ax1.set_xlabel("$\epsilon$")
    ax1.set_ylabel("relative error")
    
    ax2.legend()
    ax2.grid(which='both',linestyle=':')
    ax2.set_xlabel("$\epsilon$")
    ax2.set_ylabel("Compression Factor")
    ax1.set_xlim=ax2.get_xlim()
    
    
    fig1.savefig( 'compression_err4th.png', dpi=300, transparent=True, bbox_inches='tight' )# -*- coding: utf-8 -*-

    fig2.savefig( 'compression_rate.png', dpi=300, transparent=True, bbox_inches='tight' )
    
    return l2error,success



h5_fname = 'vorx_000060000000.h5'
wdir = '/home/phil/develop/WABBIT/'
eps_list = np.asarray([0]+[float("%1.1e" %eps) for eps in np.logspace(-5,0,14)])
wabbit_setup = {
        'mpicommand' : "mpirun --use-hwthread-cpus -np 4",
        'memory'     : "--memory=16GB"
        }
[l2error,sucess]=adaptation_for_different_eps(h5_fname, wdir, eps_list, wabbit_setup['memory'], wabbit_setup['mpicommand'], create_log_file=True)