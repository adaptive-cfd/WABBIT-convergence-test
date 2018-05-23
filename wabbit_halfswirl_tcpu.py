#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  8 13:34:58 2018

@author: engels
"""
import insect_tools
import wabbit_tools
import numpy as np
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import glob
import os
import warnings

norm = np.inf

#%%
def do_test2(rootdir, tt, name, reffile):
    dirsx = glob.glob(rootdir)
    tcpu=[]
    err=[]
    f = '/fullphi_00000'+tt+'00000.h5'
    for d in dirsx:
        if (os.path.isfile(d+f)):
            e = wabbit_tools.wabbit_error_vs_flusi( d+f, reffile, norm=norm )
            err.append(e)
            ncpu = len( glob.glob(d+'/*times.dat') )
            d = insect_tools.load_t_file(d+'/timesteps_info.t')
            tcpu.append( np.sum( d[:,1] )*float(ncpu) )
        else:
            warnings.warn("No data in that dir="+d+f)

    # sort the lists (by eps)
    err, tcpu = zip(*sorted(zip(err, tcpu )))

    plt.figure(1)
    plt.loglog( err, tcpu, label=name, marker='o')

#%%
def do_test(rootdir, name):
    dirsx = glob.glob(rootdir)
    tcpu=[]
    err=[]

    reffile='ref_flusi/T5_2048_double/flusiphi_000002500000.h5'
    f = 'fullphi_000002500000.h5'

    for d in dirsx:
        if (os.path.isfile(d+'/fullphi_000002500000.h5')):
            e = wabbit_tools.wabbit_error_vs_flusi( d+'/'+f, reffile, norm=norm )
            err.append(e)
            ncpu = len( glob.glob(d+'/*times.dat') )
            d = insect_tools.load_t_file(d+'/timesteps_info.t')
            tcpu.append( np.sum( d[:,1] )*float(ncpu) )
        else:
            warnings.warn("No data in that dir="+d+'/'+f)

    # sort the lists (by eps)
    err, tcpu = zip(*sorted(zip(err, tcpu )))

    plt.figure(1)
    plt.loglog( err, tcpu, label=name, marker='o')


#%% swirl
plt.close('all')
plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = 'serif'
plt.rcParams["font.serif"] = 'Times'

do_test("C_HALFSWIRL_newcode/adaptive_halfswirl_Bs33*JmaxInf*", "adaptive, $\\beta=10^{-2}$, $J_\mathrm{max}=14$")
do_test("C_HALFSWIRL_newcode/equidistant_halfswirl_Bs33*", "equidistant, $\\beta=10^{-2}$")

do_test("C_HALFSWIRL_newcode/adaptive_halfswirl_Bs33*Jmax4*", "adaptive, $\\beta=10^{-2}$, $J_\mathrm{max}=4$")
do_test("C_HALFSWIRL_newcode/adaptive_halfswirl_Bs33*Jmax5*", "adaptive, $\\beta=10^{-2}$, $J_\mathrm{max}=5$")
do_test("C_HALFSWIRL_newcode/adaptive_halfswirl_Bs33*Jmax6*", "adaptive, $\\beta=10^{-2}$, $J_\mathrm{max}=6$")
do_test("C_HALFSWIRL_newcode/adaptive_halfswirl_Bs33*Jmax7*", "adaptive, $\\beta=10^{-2}$, $J_\mathrm{max}=7$")

#ref='ref_flusi/T10_4096_upsampled_double/flusiphi_000005000000.h5'
#do_test2("D_HALFSWIRL_t10/adaptive*_half*", '50', "adaptive $B_s=33$, $T_a=10$, $J_\mathrm{max}=\infty$", ref)
#do_test2("D_HALFSWIRL_t10/equi*_half*", '50', "equidistant $B_s=33$, $T_a=10$", ref)

ref='ref_flusi/T5_2048_double_beta1e-4/ux_002500.h5'
do_test2("E_HALFSWIRL_beta1e-4/adaptive*_half*", '25',  "adaptive $\\beta=10^{-4}$, $J_\mathrm{max}=\infty$", ref)
do_test2("E_HALFSWIRL_beta1e-4/equi*_half*", '25',  "equidistant $\\beta=10^{-4}$", ref)

plt.figure(1)
from matplotlib.offsetbox import AnchoredText
# add annotation for metafigures
anchored_text = AnchoredText('D', loc=2, frameon=False, borderpad=0.3, pad=0, prop=dict(size=22,family='serif'))
plt.gca().add_artist(anchored_text)

a = 0.95
plt.gcf().set_size_inches( [6.71*a, 4.91*a] )
plt.gcf().subplots_adjust(top=0.91, bottom=0.13,left=0.14, right=0.97)

plt.title('8 CPU, $B_s=33$, $T_a=5$')
plt.grid()
plt.ylabel('$t_\mathrm{cpu}$')
if norm is 2:
    plt.xlabel('$||\phi-\phi_{\\mathrm{ref,flusi}}||_2/||\phi_{\\mathrm{ref,flusi}}||_2$')
elif norm is np.inf:
    plt.xlabel('$||\phi-\phi_{\\mathrm{ref,flusi}}||_\infty/||\phi_{\\mathrm{ref,flusi}}||_\infty$')
plt.legend()
plt.gcf().savefig('tcpu_vs_error_halfswirl_adaptive_vs_equi.pdf')
