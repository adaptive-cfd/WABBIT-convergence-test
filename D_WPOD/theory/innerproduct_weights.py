#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 12:05:15 2020

@author: phil
"""

import numpy as np
from numpy import sqrt
import matplotlib.pyplot as plt
import pywt
from matplotlib import rc
font = {'family' : 'serif',
        'size'   : 18}
rc('font',**font)
rc('text', usetex=True) 


def shift(arr, num, fill_value=np.nan):
    result = np.empty_like(arr)
    if num > 0:
        result[:num] = fill_value
        result[num:] = arr[:-num]
    elif num < 0:
        result[num:] = fill_value
        result[:num] = arr[-num:]
    else:
        result[:] = arr
    return result

level=10

h= np.asarray([ -1/16, 0, 9/16, 1, 9/16,0, -1/16, 0])/sqrt(2)
htilde= np.asarray([ 0, 0, 0, 1, 0, 0, 0, 0])/sqrt(2)
gtilde= np.asarray([ 0, -1/16, 0, 9/16, 1, 9/16,0, -1/16])/sqrt(2)
g= np.asarray([ 0, 0, 0, 0, 1, 0, 0, 0])/sqrt(2)

h= np.asarray([0, 1/2, 1,1/2 ,0])/sqrt(2)
htilde= np.asarray([0,  0, 1, 0, 0])/sqrt(2)
gtilde= np.asarray([0, 1/2, 1,1/2, 0])/sqrt(2)
g= np.asarray([0, 0, 1, 0, 0])/sqrt(2)


my_filter_bank = (htilde, h,  gtilde, g)
wavelet = pywt.Wavelet('My Haar Wavelet', filter_bank=my_filter_bank)
wavelet.orthogonal=True
wavelet.biorthogonal=True
[phi,psi,x] = wavelet.wavefun(level)
x = x-4


a=np.zeros(2*len(x))
taus = np.zeros(2*len(x))
delta_i=2**level
dx = x[1]-x[0]
phi_shift=[]
for tau in range(2*len(x)):
        taus[tau] = (tau-len(x)) * dx
        phi_shift = shift(phi, tau - len(x), 0)
        a[tau] = np.sum(phi*phi_shift) *dx

# a_sym = np.concatenate([np.flip(a[1:]),a])
# taus_sym = np.concatenate([-np.flip(taus[1:]),taus])
Nord = 6
M_list=[]
tau_list=[]
for i in range(Nord):
    tau_list.append(i*2**level*dx)
    phi_shift = shift(phi, i*2**level, 0)
    val = np.sum(phi*phi_shift) *dx
    M_list.append(val)

plt.close("all")
plt.figure(22)
plt.plot(x,phi,'-b')
plt.plot(x+1,phi,'--g')

plt.plot(taus,a,':k')
plt.text(1,-0.18,"$y$")
plt.xlabel("$x$")
plt.xticks(np.arange(-4,5,1))
plt.axvline(x=1, c = 'k')
plt.xlim([min(x)+1,max(x)])
plt.plot(tau_list,M_list,'ro')
plt.plot(-np.asarray(tau_list),M_list,'ro')
plt.legend(["$\phi(x)$","$\phi(x-y)$","$(\phi*\phi)(x)$"],loc='upper left')
plt.show()
#plt.grid("on")


exit


    
  
fig=plt.figure(100)
ax = fig.subplots()

for i in range(0,2*len(x),20):
    plt.gca().cla() 
    ax.plot(x,phi,'-b')
    ax.set_xlim(-4,4)
    phi_shift = shift(phi,i-len(x),0)
    ax.plot(x,phi_shift,'--g')
    ax.axvline(x=taus[i],c='k')
    ax.text(taus[i],-0.18,"$\\tau$")
    ax.plot(taus[:i],a[:i],'k:')
    plt.xlabel("$x$")
    ax.legend(["$\phi(x)$","$\phi(x-\\tau)$","$(\phi*\phi)(x)$"],loc='upper left')
    fig.canvas.draw()
    plt.pause(0.0001)
    
plt.show()