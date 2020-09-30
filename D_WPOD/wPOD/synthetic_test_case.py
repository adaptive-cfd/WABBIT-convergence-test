#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 17:04:14 2020

@author: phil
"""

import numpy as np
import matplotlib.pyplot as plt
import random
from wPOD.run_wabbit import *
from wPOD.wPODerror import *
from wPOD.adaption_test import adapt2eps
import wabbit_tools as wt
import matplotlib.pyplot as plt
import re
import os
import glob




def init(x,t,L, case="Mendez", slow_decay=True):

       N = np.asarray(np.shape(x[0]))
       Nt = len(t)
       ufield = np.zeros([np.prod(N),Nt])
       random.seed(a=1, version=2)
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
           freq_list = np.arange(0,len(x0_list)+1)
           random.shuffle(freq_list)
           if slow_decay:
               amp = [np.exp(-k/100)*np.sin(np.pi*k*t) for k in freq_list]
           else:
               amp = [np.exp(-k/3)*np.sin(np.pi*k*t) for k in freq_list]
               
                
       for i, x in enumerate(xrel_list):
           radius = np.sqrt(x[0]**2 + x[1]**2)
           phi_vec =np.reshape(fun(radius),[-1])           
           temp = np.outer(phi_vec,amp[i])
           
            
           ufield += temp
        
       # ax.legend(loc='upper left')
       # ax.spines['top'].set_visible(False)
       # ax.spines['right'].set_visible(False)
       # plt.xlabel("parameter $\mu$")
       # plt.ylabel("amplitude")
       # plt.savefig( pic_dir+"synth/acoef_data.png", dpi=300, transparent=True, bbox_inches='tight' )
       # plt.show()
       
       plt.figure(22)
       u = np.reshape(ufield,[*N,Nt])
       plt.pcolormesh(u[...,Nt//2])
       
       # plt.figure(11)
       # [U,S,V]= np.linalg.svd(ufield,0)
       # plt.semilogy(S,'*')
       # plt.xlabel("$n$")
       # plt.ylabel("$\sigma_n$")
       return ufield, amp
   

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
    t = np.linspace(0,2*np.pi,params.Nt)
    [X,Y] = np.meshgrid(*x)
    
    phi_matrix, amplitudes = init([X,Y],t,params.domain.L,case=params.case, slow_decay=params.slow_singular_value_decay)
   
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


    
    return file_list
