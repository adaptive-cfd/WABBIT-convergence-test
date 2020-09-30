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
def adapt2eps(fnames, wdir, eps_list, memory, mpicommand, normalization="Linfty", wavelets="CDF40", create_log_file=True, show_plot=True,pic_dir="./"):
    
    import re
    import os
    import glob
    import wabbit_tools as wt
    # generate new file names 
    if  not isinstance(fnames, list):
        fnames = [fnames]
    
    print( "Input data files: ", fnames, "\n")
    success=0

    folder = "./adapt_files/"
    if not os.path.exists(folder):
            os.mkdir(folder)
    os.chdir(folder)
    
    Ncomponents = len(fnames)
    sparse_files=[""] * Ncomponents
    dense_files = [""] * Ncomponents
    
    Neps    = len(eps_list)
    Nblocks = np.zeros([Neps])
    l2error = np.zeros([Neps])
    linferror = np.zeros([Neps])
    
    for i, eps in enumerate(eps_list):
  
        for k, fname in enumerate(fnames):
            path, file = os.path.split(fname)
            filesparse =file.replace('_','-eps_%1.1e_'%eps)
            filedense =file.replace('_','-dense_')
            os.system("cp "+path+"/"+file+ " ./"+filedense )
            os.system("cp "+path+"/"+file+ " ./"+filesparse)
            sparse_files[k] = filesparse
            dense_files[k] =  filedense
            
        # command for sparsing file
        command = mpicommand + " " +  wdir + \
                 "wabbit-post --dense-to-sparse --eps=%1.1e --order="+wavelets+" --eps-norm="+normalization +" " + memory + \
                  ' --files="'+ " ".join(sparse_files[p] for p in range(Ncomponents))+ '" ' 

        c       = command % (eps)  # change eps
        if create_log_file:
            c += " >> adapt.log" 
        # -----------------------------
        # Execute Command
        # -----------------------------
        print("\n\n###################################################################")
        print("\t\teps =%1.1e"%eps)
        print("###################################################################")
        print("\n",c,"\n\n")
        success += os.system(c)
        for filesparse,filedense in zip(sparse_files,dense_files):
            dc = mpicommand + " " +  wdir + \
                 "wabbit-post --sparse-to-dense " + \
                  " "+ filesparse + " "+filedense     
            if create_log_file:
                dc += " >> adapt.log" 
            print("\n",dc,"\n\n")
            success += os.system(dc) # dense the file for comparison to original
            if success != 0:
                print("command did not execute successfully")
                break
            
        l2error[i] += wt.wabbit_error_vs_wabbit(fnames,dense_files, norm=2, dim=2)  
        #  l2error[i] = wt.wabbit_error_vs_(fname_ref[0],fname_dense[0], norm=2, dim=2)   
        linferror[i] += wt.wabbit_error_vs_wabbit(fnames,dense_files, norm=np.inf, dim=2)   

            # compute compression
        Nblocks[i]=sum(wt.block_level_distribution_file( sparse_files[0] ))
            

     # compute number of dense blokcs
    Nblocksdense = sum(wt.block_level_distribution_file( dense_files[0] ))
            

    os.chdir("../")
    
  
    if show_plot:
        ##############################################################################
        #    Plot
        ##############################################################################
        #### plot Lp error    
        fig1, ax1 = plt.subplots() 
        fig2, ax2 = plt.subplots() 
        Ne = 0
        l2plt, = ax1.loglog(eps_list[Ne:],l2error[Ne:],'-o',label="$\Vert u(x) - [u(x)]^\epsilon \Vert_2$")
        #linfplt, = ax1.loglog(eps_list[Ne:],linferror[Ne:],'-.*', label= "$\Vert u(x) - [u(x)]^\epsilon \Vert_\infty$")
        ####  plot compression rate
        ax2.semilogx(eps_list,Nblocks/Nblocksdense,'-o')
        # Create a legend for the first line.
        ax1.loglog(eps_list,eps_list, 'k--',label=r"$\epsilon$")
        ax1.grid(which='both',linestyle=':')
        ax1.set_xlabel("$\epsilon$")
        ax1.set_ylabel("relative error")
        ax1.legend()
        
        #ax2.legend()
        ax2.grid(which='both',linestyle=':')
        ax2.set_xlabel("$\epsilon$")
        ax2.set_ylabel("Compression Factor $C_{\mathrm{f}}$")
        ax1.set_xlim=ax2.get_xlim()
        
        
        ####  plot compression rate vs error
        fig3, ax3 = plt.subplots() 
        ax3.semilogy(Nblocks/Nblocksdense, l2error,'-o')
        # Create a legend for the first line.
        
        #ax2.legend()
        ax3.grid(which='both',linestyle=':')
        ax3.set_ylabel("$\Vert u(x) - [u(x)]^\epsilon \Vert_2$")
        ax3.set_xlabel("Compression Factor $C_{\mathrm{f}}$")
        
        
        
        
        fig1.savefig( 'compression_err4th.png', dpi=300, transparent=True, bbox_inches='tight' )# -*- coding: utf-8 -*-
    
        fig2.savefig( 'compression_rate.png', dpi=300, transparent=True, bbox_inches='tight' )
    
        fig3.savefig( 'compression_err_vs_rate.png', dpi=300, transparent=True, bbox_inches='tight' )# -*- coding: utf-8 -*-
        
        
        fig1.savefig(pic_dir+ 'compression_err4th.svg', dpi=300, transparent=True, bbox_inches='tight' )# -*- coding: utf-8 -*-
        fig2.savefig(pic_dir+ 'compression_rate.svg', dpi=300, transparent=True, bbox_inches='tight' )  
        fig3.savefig(pic_dir+ 'compression_err_vs_rate.svg', dpi=300, transparent=True, bbox_inches='tight' )  

    return l2error, linferror, Nblocks, Nblocksdense



# h5_fname = '/home/phil/develop/WPOD/D_WPOD/bump/01/Jmax5/u_000006283185.h5'
# wdir = '/home/phil/develop/WABBIT/'
# eps_list = np.asarray([0]+[float("%1.1e" %eps) for eps in np.logspace(-4,0,10)])
# wabbit_setup = {
#          'mpicommand' : "mpirun --use-hwthread-cpus -np 4",
#          'memory'     : "--memory=16GB"
#          }
# [l2error,sucess]=adapt2eps(h5_fname, wdir, eps_list, wabbit_setup['memory'], wabbit_setup['mpicommand'], create_log_file=True)