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

plt.close('all')
plt.figure()

files = glob.glob('compression_test/phi_*e-*.h5')

for norm in [2, np.inf]:
    e = []
    eps = []

    for file in files:
        e.append( wabbit_tools.wabbit_error('./', show=False, norm=norm, file=file) )

        n = file.replace('compression_test/phi_','' )
        n = n.replace('.h5','')

        eps.append( float(n) )


    eps, e = zip(*sorted(zip(eps, e)))


    plt.rcParams["text.usetex"] = True
    plt.rcParams["font.family"] = 'serif'
    plt.rcParams["font.serif"] = 'Times'
    plt.loglog( eps, e, label="4th"+" [%2.2f]" % (wabbit_tools.convergence_order(eps,e)), marker='o')
    plt.loglog( eps, eps, linestyle='--', color='k' )

    plt.xlabel('$\\varepsilon$')
    if norm is 2:
        plt.ylabel('$||\phi-\phi_{\\mathrm{ex}}||_2/||\phi_{\\mathrm{ex}}||_2$')
    else:
        plt.ylabel('$||\phi-\phi_{\\mathrm{ex}}||_\infty/||\phi_{\\mathrm{ex}}||_\infty$')