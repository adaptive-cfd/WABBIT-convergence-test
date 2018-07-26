#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 18:17:56 2017

@author: engels
"""

import numpy as np
import matplotlib.pyplot as plt
import glob
import wabbit_tools
import os

#%%
def equidistant(rootdir, title, norm, ax):
    dirsx = glob.glob( rootdir+'*' )
    Nblocks, e = [], []
    ref = 'ref_flusi/T5_2048_double/flusiphi_000002500000.h5'

    if len(dirsx)==0:
        raise ValueError('no data')

    for d in dirsx:
        if (os.path.isfile(d+'/fullphi_000002500000.h5')):
            err = wabbit_tools.wabbit_error_vs_flusi( d+'/fullphi_000002500000.h5', ref, norm=norm )
            N, Bs = wabbit_tools.fetch_Nblocks_dir(d, return_Bs=True)

            # append to plot-lists
            Nblocks.append( (Bs-1)*np.sqrt(N) )
            e.append( err )

    Nblocks, e = zip(*sorted(zip(Nblocks, e)))

    ax.loglog( Nblocks, e, label=title+" [%2.2f]" % (wabbit_tools.convergence_order(Nblocks,e)), marker='o')

#%%
def adaptive(rootdir, title, norm, Bs, ax):
    dirsx = glob.glob( rootdir )
    e, Nb = [], []
    ref = 'ref_flusi/T5_2048_double/flusiphi_000002500000.h5'

    if len(dirsx)==0:
        raise ValueError('no data')

    for d in dirsx:
        if (os.path.isfile(d+'/fullphi_000002500000.h5')):
            # compute error
            err = wabbit_tools.wabbit_error_vs_flusi( d+'/fullphi_000002500000.h5', ref, norm=norm )
            e.append( err )

            # compute number of points
            N, Bs = wabbit_tools.fetch_Nblocks_RHS_dir(d, return_Bs=True)
            N2, Bs = wabbit_tools.fetch_Nblocks_dir(d, return_Bs=True)

            print("N1=%i N2=%i" % (N,N2))

            Nb.append( (Bs-1)*np.sqrt(N) )
    Nb, e = zip(*sorted(zip(Nb, e)))

    plt.loglog( Nb, e, label=title+" [%2.2f]" % (wabbit_tools.convergence_order(Nb,e)), marker='o')

#%%
def adaptive_epsopt(dirlist, title, norm, Bs, ax):
    e, Nb = [], []
    ref = 'ref_flusi/T5_2048_double/flusiphi_000002500000.h5'

    if len(dirlist)==0:
        raise ValueError('no data')

    for d in dirlist:
        if (os.path.isfile(d+'/fullphi_000002500000.h5')):
            # compute error
            err = wabbit_tools.wabbit_error_vs_flusi( d+'/fullphi_000002500000.h5', ref, norm=norm )
            e.append( err )

            # compute number of points
            N, Bs = wabbit_tools.fetch_Nblocks_RHS_dir(d, return_Bs=True)
            Nb.append( (Bs-1)*np.sqrt(N) )

    # sort data by number of blocks
    Nb, e = zip(*sorted(zip(Nb, e)))
    plt.loglog( Nb, e, label=title+" [%2.2f]" % (wabbit_tools.convergence_order(Nb,e)), marker='o')

#%%
plt.close('all')
plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = 'serif'
plt.rcParams["font.serif"] = 'Times'

plt.figure(1)
ax = plt.gca()

equidistant('C_HALFSWIRL_newcode/equidistant_halfswirl_Bs33','equidistant $B_s=33$', np.inf, ax)
adaptive('C_HALFSWIRL_newcode/adaptive_halfswirl_Bs33*Jmax?_eps1.00000000e-07','$\\varepsilon=10^{-7}$', np.inf, 33, ax)
#adaptive_epsopt(['C_HALFSWIRL_newcode/adaptive_halfswirl_Bs33_Jmax3_eps5.45559478e-04',
#                 'C_HALFSWIRL_newcode/adaptive_halfswirl_Bs33_Jmax4_eps7.84759970e-05',
#                 'C_HALFSWIRL_newcode/adaptive_halfswirl_Bs33_Jmax5_eps6.95192796e-06',
#                 'C_HALFSWIRL_newcode/adaptive_halfswirl_Bs33_Jmax6_eps1.00000000e-06'],
#                '$\\varepsilon=\\varepsilon_{\mathrm{opt}}$', np.inf, 33, ax)


from matplotlib.offsetbox import AnchoredText
# add annotation for metafigures
anchored_text = AnchoredText('B', loc=2, frameon=False, borderpad=0.3, pad=0, prop=dict(size=22,family='serif'))
ax.add_artist(anchored_text)

a = 0.75
plt.gcf().set_size_inches( [6.71*a, 4.91*a] )
plt.gcf().subplots_adjust(top=0.91, bottom=0.13,left=0.14, right=0.97)

plt.xlim([1e2, 1e4])
plt.title('$T_{a}=5$, $\\beta=10^{-2}$, $B_s =33$')
plt.grid()
plt.legend() #loc='upper left', bbox_to_anchor=(0,1.02,1,0.2), prop={'size': 6})
plt.xlabel('$N_{y}=(B_{s}-1)\\sqrt{N_{b}}$')

#plt.ylabel('$||\phi-\phi_{\\mathrm{ref,flusi}}||_2/||\phi_{\\mathrm{ref,flusi}}||_2$')
plt.ylabel('$||\phi-\phi_{\\mathrm{ref,flusi}}||_\infty/||\phi_{\\mathrm{ref,flusi}}||_\infty$')

plt.gcf().savefig('convergence_dx_adaptive_eps1e-7_halfswirl.pdf')
