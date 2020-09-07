#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 17:06:01 2020

@author: phil
"""

from wPOD.randomSVD import rSVD
from wPOD.synthetic_test_case import synthetic_test_case, init
import numpy as np
from numpy.linalg import svd
import matplotlib.pyplot as plt
import time

# %%

class params:
   case = "Bumbs" # choose [Bumbs,Mendez] 
   slow_singular_value_decay=False
   #eps_list  = np.asarray([0]+[float("%1.1e" %eps) for eps in np.logspace(-12,0,6)])  # threshold of adaptation
   eps_list  = np.asarray([0]+[float("%1.1e" %eps) for eps in np.logspace(-8,0,10)])  # threshold of adaptation
   jmax_list = [4,5]        # maximal tree level
   Nt= 2**7
   target_rank = 100
   
   class domain:
        N = [2**7, 2**7]      # number of points
        L = [30, 30]            # leng
        
dim = len(params.domain.L)
x = [np.linspace(0,params.domain.L[d],params.domain.N[d]) \
     for d in range(dim) ]
t = np.linspace(0,2*np.pi,params.Nt)
[X,Y] = np.meshgrid(*x)

phi_matrix, amplitudes = init([X,Y],t,params.domain.L,case=params.case, slow_decay=params.slow_singular_value_decay)
   
phi = np.reshape(phi_matrix,[*params.domain.N,params.Nt])

Nsnapshots = phi_matrix.shape[1]

ts = time.time()
[U, S, VT] = svd(phi_matrix,0)
t_svd = time.time() - ts
print("svd time: %2.2f sec"% t_svd)

# %%

rank = 50
nr_measurements = rank
nr_power_iter = 0
nr_oversampl  = 10

ts = time.time()
[rU,rS,rVT,ndat] = rSVD(phi_matrix, r = nr_measurements , 
                   q = nr_power_iter, p = nr_oversampl)
t_rsvd = time.time() - ts
print("rsvd time: %2.2f sec"% t_rsvd)

error_rsvd=[np.linalg.norm(phi_matrix-rU[:,:r]@np.diag(rS[:r])@rVT[:r,:],ord=2)/S[0] for r in range(rank)]
plt.figure(94)

plt.semilogy(S/S[0],'-*')
plt.semilogy(error_rsvd,'x')
plt.semilogy(rS/S[0],'o')
