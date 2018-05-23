#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 19 09:09:11 2018

@author: engels
"""

import glob
import wabbit_tools
import os.path
import matplotlib.pyplot as plt
import farge_colormaps
import insect_tools

fig = plt.figure(2)
a = 2
plt.gcf().set_size_inches( [5.0*a, 5.0*a] )
plt.gcf().subplots_adjust(top=0.95, bottom=0.05,left=0.02, right=0.97)

##for prefix, cmap in zip(['vor','p','ux','uy','div'],['vorticity','pressure','velocity','velocity','streamfunction']):
#for prefix, cmap in zip(['mask'],['jet']):
#    files = glob.glob('/home/engels/dev/WABBIT4-new-physics/cyl/'+prefix+'_*.h5')
#    files.sort()
#
#    if cmap is 'jet':
#        fc = 'jet'
#    else:
#        fc = farge_colormaps.farge_colormap_multi( taille=600, limite_faible_fort=0.2, etalement_du_zero=0.02, type=cmap )
#
#    for file, i in zip(files[1:], range(len(files[1:]))):
#        wabbit_tools.plot_wabbit_file( file, savepng=True, cmap=fc, ticks=False, colorbar=True, dpi=300, title=True,
#                                      block_edge_color='w', caxis_symmetric=False, shading='gouraud' )


#%% flusi cylinder
for prefix, cmap in zip(['vorx','p','uy','uz','mask'],['vorticity','pressure','velocity','velocity','streamfunction']):
    fname = '/home/engels/dev/FLUSI/cyl/'+prefix+'_001500.h5'

    plt.gcf().clf()

    # read data
    time, box, origin, data = insect_tools.read_flusi_HDF5(fname)

    # colormap
    fc = farge_colormaps.farge_colormap_multi( taille=600, limite_faible_fort=0.2, etalement_du_zero=0.02, type=cmap )

    hplot = plt.pcolormesh( data[0,:,:].transpose(), cmap=fc, shading='flat' )

    # get color limits
    a = hplot.get_clim()

    # set symmetric color limit
    c = min( [abs(a[0]), a[1]] )
    hplot.set_clim( (-c,c)  )

    # use rasterization for the patch we just draw
    hplot.set_rasterized(True)
    plt.colorbar()


    plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom='off',      # ticks along the bottom edge are off
    top='off',         # ticks along the top edge are off
    labelbottom='off') # labels along the bottom edge are off

    plt.tick_params(
    axis='y',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom='left',      # ticks along the bottom edge are off
    top='right',         # ticks along the top edge are off
    labelleft='off') # labels along the bottom edge are off

    plt.axis('tight')
    plt.axes().set_aspect('equal')
    plt.gcf().canvas.draw()
    plt.savefig( fname.replace('.h5','.png') )