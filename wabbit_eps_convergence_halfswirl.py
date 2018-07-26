#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 18:17:56 2017

@author: engels
"""

import wabbit_tools
import numpy as np
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import glob
import os

norm = np.inf

def do_test(rootdir, J, name, ax1, ax2):
    dirsx = glob.glob(rootdir+'*'+'Jmax'+J+'*')
    EPS=[]
    err=[]
    Nblocks=[]

    reffile='ref_flusi/T5_2048_double/flusiphi_000002500000.h5'

    for d in dirsx:
        if (os.path.isfile(d+'/fullphi_000002500000.h5')):
            e = wabbit_tools.wabbit_error_vs_flusi( d+'/fullphi_000002500000.h5', reffile, norm=norm )
            err.append(e)
            Nblocks.append( wabbit_tools.fetch_compression_rate_dir(d) )
#            Nblocks.append( wabbit_tools.fetch_Nblocks_dir(d) )
            EPS.append( wabbit_tools.fetch_eps_dir(d) )

    # sort the lists (by eps)
    EPS, err, Nblocks = zip(*sorted(zip(EPS, err, Nblocks)))
#    name = name +" [%2.2f]" % (wabbit_tools.convergence_order(EPS,err))


    ax1.loglog( EPS, err, label=name, marker='o')
    ax2.semilogx( EPS, Nblocks, label=name, marker='o')


#%%
plt.close('all')
plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = 'serif'
plt.rcParams["font.serif"] = 'Times'

plt.figure(1)
ax1 = plt.gca()
plt.figure(2)
ax2 = plt.gca()

do_test("C_HALFSWIRL_newcode/adaptive_halfswirl_Bs33", "Inf", "$J_\mathrm{max}=14$",ax1 , ax2)
do_test("C_HALFSWIRL_newcode/adaptive_halfswirl_Bs33", "7", "$J_\mathrm{max}=7$",ax1 , ax2)
do_test("C_HALFSWIRL_newcode/adaptive_halfswirl_Bs33", "6", "$J_\mathrm{max}=6$",ax1 , ax2)
do_test("C_HALFSWIRL_newcode/adaptive_halfswirl_Bs33", "5", "$J_\mathrm{max}=5$",ax1 , ax2)
do_test("C_HALFSWIRL_newcode/adaptive_halfswirl_Bs33", "4", "$J_\mathrm{max}=4$",ax1 , ax2)
do_test("C_HALFSWIRL_newcode/adaptive_halfswirl_Bs33", "3", "$J_\mathrm{max}=3$",ax1 , ax2)


plt.figure(1)
a = 0.75
plt.gcf().set_size_inches( [6.71*a, 4.91*a] )
plt.gcf().subplots_adjust(top=0.91, bottom=0.13,left=0.14, right=0.97)
plt.title('$T_{a}=5$, $\\beta=10^{-2}$, $B_s =33$')
plt.grid()
plt.xlabel('$\\varepsilon$')
plt.ylabel('$||\phi-\phi_{\\mathrm{ex}}||_\infty/||\phi_{\\mathrm{ex}}||_\infty$')
plt.legend()

from matplotlib.offsetbox import AnchoredText
# add annotation for metafigures
anchored_text = AnchoredText('A', loc=2, frameon=False, borderpad=0.3, pad=0, prop=dict(size=22,family='serif'))
ax1.add_artist(anchored_text)

plt.gcf().savefig('halfswirl-adaptive-eps-bs33.pdf')


plt.figure(2)
a = 0.75
plt.gcf().set_size_inches( [6.71*a, 4.91*a] )
plt.gcf().subplots_adjust( top=0.91, bottom=0.13, left=0.14, right=0.97 )

plt.title('$T_{a}=5$, $\\beta=10^{-2}$, $B_s =33$')
plt.grid()
plt.xlabel('$\\varepsilon$')
plt.ylabel('compression rate')
plt.legend()

from matplotlib.offsetbox import AnchoredText
# add annotation for metafigures
anchored_text = AnchoredText('C', loc=2, frameon=False, borderpad=0.3, pad=0, prop=dict(size=22,family='serif'))
ax2.add_artist(anchored_text)

plt.gcf().savefig('halfswirl-adaptive-nblocks-bs33.pdf')