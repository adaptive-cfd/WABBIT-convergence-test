#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 10:55:09 2019

@author: krah
"""
import numpy as np
import random
import wabbit_tools as wt
import matplotlib.pyplot as plt
import re
import os
import glob
import farge_colormaps
from run_wabbit import *
from wPODerror import *
import farge_colormaps
from adaption_test import adaptation_for_different_eps
from matplotlib import rc
###############################################################################
# LATEX FONT
font = {'family' : 'serif',
        'size'   : 16}
rc('font',**font)
rc('text', usetex=True)   
###############################################################################
###############################################################################
# COLORMAP:
fc = farge_colormaps.farge_colormap_multi( taille=600,limite_faible_fort=0.2, etalement_du_zero=0.04, type='vorticity' )
###############################################################################
###############################################################################
# %% Change parameters here
###############################################################################
# directories needed
dirs= {
       'wabbit' : "~/develop/WABBIT/" ,         # directory of wabbit
       'work'   : "/home/krah/develop/WPOD/D_WPOD/bumb",                         # where to save big data files
       'images' : "/home/krah/paper/01_paper/01_wPOD/figures/"           
       }

# setup for wabbit call
wabbit_setup = {
        'mpicommand' : "/usr/lib64/mpi/gcc/mpich/bin/mpirun -np 6",
        'memory'     : "--memory=80GB"
        }

pic_dir = dirs["images"]
# parameters to adjust
class params:
   case = "Bumbs" # choose [Bumbs,Mendez] 
   eps_list  = np.asarray([0]+[float("%1.1e" %eps) for eps in np.logspace(-12,0,6)])  # threshold of adaptation
   jmax_list = [6]        # maximal tree level
   Nt= 256
   
   class domain:
        N = [2**11, 2**11]            # number of points for 3d use 3 elements
        L = [40, 40]            # length of domain
  
   #exp(-(x²+y²)/2/sigma²)
   def init(x,t,L, case="Mendez", slow_decay=True):

       N = np.asarray(np.shape(x[0]))
       Nt = len(t)
       ufield = np.zeros([np.prod(N),Nt])
       
       if case =="Mendez":  
           sigma = 1
           # gauss function
           fun = lambda r: np.exp(-r**2/(2*sigma**2))/(2*np.pi*sigma**2)
           Nt = len(t)
           f = [15.0, 0.1, 7.0]
           x0_list = [np.asarray([30,10]),np.asarray([10,30]), np.asarray([20,20])]
           xrel_list  = [np.asarray([x-x0 for x,x0 in zip(x,x0)]) for x0 in x0_list]
           
           phi = np.zeros([3,*N])
           amp = [np.sin(2*np.pi*f[0]*t)*(np.tanh((t-2)*10)-np.tanh((t-4)*10)), 
                  np.sin(2*np.pi*f[1]*t- np.pi/3), 
                  np.sin(2*np.pi*f[2]*t)*(t - 2.55)**2]
           
               
       elif case == "Bumbs":
           # define bumb function
           fun = lambda r: np.where(np.abs(r)<1,np.exp(-1/(1-r**2)),0)
               
           Nr_of_bumbs = [ int(l // 2)  for l in L]
           delta = [2, 2 ] # horizontal and vertical distance of bumbs
           #phi = np.zeros([Nr_of_bumbs[0]*Nr_of_bumbs[1],*N])
           if any(Nr_of_bumbs)<1: 
               print("Domain must be larger then 2") 
               return
       
           x0_list = [np.asarray([(0.5+ix)*delta[0],(0.5+iy)*delta[0]]) for ix in range(Nr_of_bumbs[0]) for iy in range(Nr_of_bumbs[1])]
           xrel_list  = [np.asarray([x-x0 for x,x0 in zip(x,x0)]) for x0 in x0_list]
           freq_list = np.arange(0,len(x0_list)+1)+1
           random.shuffle(freq_list)
           if slow_decay:
               amp = [np.sin(np.pi*k*t)/np.sqrt(np.pi) for k in freq_list]
           else:
               amp = [np.exp(-k/3)*np.sin(np.pi*k*t)/np.sqrt(np.pi) for k in freq_list]
               
       fig = plt.figure(55)
       ax = fig.subplots()
        
       for i, x in enumerate(xrel_list):
           radius = np.sqrt(x[0]**2 + x[1]**2)
           phi_vec =np.reshape(fun(radius),[-1])           
           temp = np.outer(phi_vec,amp[i])
           #norm = np.linalg.norm(temp,ord=2)
           #temp *= 1/norm
            
           #amp[i] *=1/norm
            
            # plot temporal amplitude
           ax.plot(t,amp[i],label="$a_{"+str(i+1)+"}(\mu)$")
            
           ufield += temp
     
       ax.legend(loc='upper left')
       ax.spines['top'].set_visible(False)
       ax.spines['right'].set_visible(False)
       plt.xlabel("parameter $\mu$")
       plt.ylabel("amplitude")
       plt.savefig( pic_dir+"synth/acoef_data.png", dpi=300, transparent=True, bbox_inches='tight' )
       plt.show()
       
       plt.figure(22)
       u = np.reshape(ufield,[*N,Nt])
       plt.pcolormesh(u[...,Nt//2])
       
       
       plt.figure(11)
       [U,S,V]= np.linalg.svd(ufield,0)
       plt.semilogy(S,'*')
       plt.xlabel("$n$")
       plt.ylabel("$\sigma_n$")
       return ufield, amp

###############################################################################
# %% 
###############################################################################



def synthetic_test_case(dirs, params, wabbit_setup):
    
    qname="u"
    mpicommand = wabbit_setup["mpicommand"]
    memory     = wabbit_setup["memory"]
    wdir = dirs["wabbit"]
    work = dirs["work"]
    jmax_list = params.jmax_list
    # set initial gird
    dim = len(params.domain.L)
    x = [np.linspace(0,params.domain.L[d],params.domain.N[d]) \
         for d in range(dim) ]
    t = np.linspace(0,6,params.Nt)
    [X,Y] = np.meshgrid(*x)
    
    phi_matrix, amplitudes = params.init([X,Y],t,params.domain.L,case=params.case)
   
    phi = np.reshape(phi_matrix,[*params.domain.N,params.Nt])
    
    for jmax in jmax_list:
        
        directory = work+"/Jmax"+str(jmax)+"/"
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        
        Bs = wt.field_shape_to_bs(params.domain.N,jmax)
    
        # write snapshotmatrix:
        file_list = []
        for it, time in enumerate(t):
            file_ref = wt.dense_to_wabbit_hdf5(phi[...,it],directory+qname, Bs, params.domain.L,time=time)
            file_list.append(file_ref)
    
    
    folder = os.path.split(os.path.split(file_ref)[0])[0]
    # params.Bs_list = []
    # l2error=np.zeros([len(params.jmax_list),len(params.eps_list)])
    # linferror=np.zeros_like(l2error)
    # compress = np.zeros_like(l2error)
    
    data = {"folder" :  folder,
        "qname" : [qname]
        }
    mode_lists = ["mode1_list.txt"]
    run_wabbit_POD(wabbit_setup, dirs, data, params.jmax_list, params.eps_list, mode_lists, reconstructed_iteration=10)
    #     Bs = wt.field_shape_to_bs(params.domain.N,jmax)
    #     params.Bs_list.append(Bs)        
    #     # adapt field for different eps using wabbit-post -dense-to-sparse
    #     print("\n\n###################################################################")
    #     print("###################################################################")
    #     print( "\t\tJmax: ", jmax, "\n\n")
    #     for k,eps in enumerate(params.eps_list):
    #         # create dense field and save to work
    #         fname = "phi-j"+str(jmax)+"-eps"
    #         file = work +"/"+fname
    #         file = wt.dense_to_wabbit_hdf5(phi,file, Bs, params.domain.L,eps*100)
    #         command = mpicommand + " " +  wdir + \
    #              "wabbit-post --dense-to-sparse --eps=" + str(eps) + " --order=CDF44 --eps-norm=Linfty " + memory + \
    #               " --files="+ file + ">adapt.log"
    #               # command for densing file again
    #         dense_command = mpicommand + " " +  wdir + \
    #              "wabbit-post --sparse-to-dense " + \
    #               " "+ file + " "+file + ">>adapt.log"
    #         # -----------------------------
    #         # Execute Command
    #         # -----------------------------
    #         print("\n\n###################################################################")
    #         print("\t\teps =",eps)
    #         print("###################################################################")
    #         print("\n",command,"\n\n")
    #         success = os.system(command)   # execute command
    #         if np.mod(k,4)==0:
    #             wt.plot_wabbit_file(file,savepng=True)
    #         compress[j,k] = sum(wt.block_level_distribution_file( file ))
             
    #         print("\n",dense_command,"\n\n")
    #         success += os.system(dense_command) # dense the file for comparison to original
    #         if success != 0:
    #             print("command did not execute successfully")
    #             return
            
    #         # ---------------------------
    #         # compare to original file
    #         # --------------------------
    #         l2error[j,k] = wt.wabbit_error_vs_wabbit(file_ref,file, norm=2, dim=2)  
    #         linferror[j,k] = wt.wabbit_error_vs_wabbit(file_ref,file, norm=np.inf, dim=2)  
    #         compress[j,k] /=  sum(wt.block_level_distribution_file( file ))
        
    #         # delete file
    #         os.system("rm "+ file)
    
    # # delete reference file
    # os.system("rm " + file_ref)
    return file_list
 
# %%

Jmax_dir_list = [ "Jmax"+str(Jmax)+"/" for Jmax in params.jmax_list]
eps_list = params.eps_list
eps_dir_list = [ "eps%1.1e" %eps for eps in eps_list]

file_list = synthetic_test_case(dirs, params, wabbit_setup)
delta_err,clist=plot_wPODerror(params.jmax_list, Jmax_dir_list,eps_dir_list,eps_list, dirs, n_star=1)

h5_fname = file_list[params.Nt//2]
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
