#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 15:21:45 2020

@author: phil
"""

import sympy as sym
import numpy as np
import sys
import mpmath
sys.modules['sympy.mpmath'] = mpmath
from sympy import Function,Symbol
from sympy.mpmath import nprod

x = sym.Symbol('x')

y = sym.Symbol('y')

# %% Define the Lagrange Polynoms of order Nord with points 
# x_i = i-Nord//2+1   for i = 0, ..., Nord

Nord = 4                                 # order of langrange polynomials 
index = list(range(-Nord//2+1,Nord//2+1))  # list of gridpoints
L = []                                     # list of polynomials in the basis
for i in index:
    l = 1
    for m in index:
        if m != i:
            l *= (x-m)/(i-m)
            
    L.append(l)

# filter coefficients h_k = L_k (0.5)
filter_coef = [l.subs(x,0.5) for l in L]
filter_coef = np.round(np.asarray(filter_coef,dtype=np.float32),10)
print( "wavelet filter coefficients, see table 2 in wPOD paper:\n\n h_k =" ,
      filter_coef)
# %%
    