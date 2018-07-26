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

norm = 2

#%%
def do_test(rootdir, J, name, factor=1):
    dirsx = glob.glob(rootdir+'*'+'Jmax'+J+'*')
    plt.figure(1)
    for d in dirsx:
        d = insect_tools.load_t_file(d+'/timesteps_info.t')
        if factor == 4:
            plt.loglog( d[:,3]*factor, d[:,1], label=name, marker='o', linestyle='None', color=[0.5,0.5,0.5])
        else:
            plt.loglog( d[:,3]*factor, d[:,1], label=name, marker='o', linestyle='None')


#%% swirl
plt.close('all')
plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = 'serif'
plt.rcParams["font.serif"] = 'Times'

#do_test("C_HALFSWIRL_newcode/adaptive_halfswirl_Bs17", "4", "Bs=17 (T=5.0, Jmax=4)")
#do_test("C_HALFSWIRL_newcode/adaptive_halfswirl_Bs17", "5", "4th-4th-4th (T=5.0, Jmax=5)")
#do_test("C_HALFSWIRL_newcode/adaptive_halfswirl_Bs17", "6", "4th-4th-4th (T=5.0, Jmax=6)")
#do_test("C_HALFSWIRL_newcode/adaptive_halfswirl_Bs17", "7", "4th-4th-4th (T=5.0, Jmax=7)")
#do_test("C_HALFSWIRL_newcode/adaptive_halfswirl_Bs17", "Inf", "4th-4th-4th (T=5.0, Jmax=Inf)")

do_test("C_HALFSWIRL_newcode/adaptive_halfswirl_Bs33", "Inf", "Bs=33 (T=5.0, Jmax=Inf)", 4)
do_test("C_HALFSWIRL_newcode/equidistant_halfswirl_Bs33", "", "equi, bs=33 (T=5.0)", 1)

#do_test("C_HALFSWIRL_newcode/equidistant_halfswirl_Bs17", "", "equi, bs=17 (T=5.0)")
#do_test("C_HALFSWIRL_newcode/equidistant_halfswirl_Bs65", "", "equi, bs=65 (T=5.0)")

#do_test2("D_HALFSWIRL_t10/adaptive", "", "adaptive, bs=33 (T10.0)")
#do_test2("D_HALFSWIRL_t10/equi", "", "equdistant, bs=33 (T=10.0)")


plt.figure(1)
plt.title('Half Swirl test')
plt.grid()
plt.ylabel('t_cpu')
plt.xlabel('N blocks RHS')
plt.gcf().subplots_adjust(top=0.82)
plt.gcf().savefig('halfswirl_nblocksRHS_tcpu.pdf')
