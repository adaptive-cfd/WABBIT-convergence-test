#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 17:06:01 2020

@author: phil
"""

from randomSVD import rSVD
from synthetic_test_params import params
import numpy as np
from numpy.linalg import svd
import matplotlib.pyplot as plt
import time

# %%
dim = len(params.domain.L)
x = [np.linspace(0,params.domain.L[d],params.domain.N[d]) \
     for d in range(dim) ]
t = np.linspace(0,2*np.pi,params.Nt)
[X,Y] = np.meshgrid(*x)

phi_matrix, amplitudes = params.init([X,Y],t,params.domain.L,case=params.case, slow_decay=False)
   
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
[rU,rS,rVT] = rSVD(phi_matrix, r = nr_measurements , 
                   q = nr_power_iter, p = nr_oversampl)
t_rsvd = time.time() - ts
print("rsvd time: %2.2f sec"% t_rsvd)

error_rsvd=[np.linalg.norm(phi_matrix-rU[:,:r]@np.diag(rS[:r])@rVT[:r,:],ord=2)/S[0] for r in range(rank)]
plt.figure(94)

plt.semilogy(S/S[0],'-*')
plt.semilogy(error_rsvd,'x')
plt.semilogy(rS/S[0],'o')
