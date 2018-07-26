#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 29 15:43:42 2017

@author: engels
"""

import glob
import wabbit_tools
import os.path
import matplotlib.pyplot as plt


#wabbit_tools.plot_wabbit_file( 'A_CONV/dx1_nonequi_4th-4th-4th_0/phi_000000000000.h5', savepng=True, cmap='rainbow', caxis=[0,1] )
#wabbit_tools.plot_wabbit_file( 'A_CONV/dx2_nonequi_2nd-2nd-4th_0/phi_000000000000.h5', savepng=True, cmap='rainbow', caxis=[0,1] )
#
#wabbit_tools.plot_wabbit_file( 'A_CONV/adapt1_0_1.00000000e-07/phi_000000000000.h5', savepng=True, cmap='rainbow', caxis=[0,1] )
#wabbit_tools.plot_wabbit_file( 'A_CONV/adapt1_0_1.00000000e-07/phi_000000500000.h5', savepng=True, cmap='rainbow', caxis=[0,1] )
#wabbit_tools.plot_wabbit_file( 'A_CONV/adapt1_0_1.00000000e-07/phi_000001000000.h5', savepng=True, cmap='rainbow', caxis=[0,1] )
#
#wabbit_tools.plot_wabbit_file( 'A_CONV/adapt1_17_1.51177507e-05/phi_000000000000.h5', savepng=True, cmap='rainbow', caxis=[0,1] )
#wabbit_tools.plot_wabbit_file( 'A_CONV/adapt1_17_1.51177507e-05/phi_000000500000.h5', savepng=True, cmap='rainbow', caxis=[0,1] )
#wabbit_tools.plot_wabbit_file( 'A_CONV/adapt1_17_1.51177507e-05/phi_000001000000.h5', savepng=True, cmap='rainbow', caxis=[0,1] )
#
#wabbit_tools.plot_wabbit_file( 'A_CONV/adapt2_0_2.06913808e-05/phi_000000000000.h5', savepng=True, cmap='rainbow', caxis=[0,1] )
#wabbit_tools.plot_wabbit_file( 'A_CONV/adapt2_0_2.06913808e-05/phi_000000500000.h5', savepng=True, cmap='rainbow', caxis=[0,1] )
#wabbit_tools.plot_wabbit_file( 'A_CONV/adapt2_0_2.06913808e-05/phi_000001000000.h5', savepng=True, cmap='rainbow', caxis=[0,1] )
#
#
#wabbit_tools.plot_wabbit_file( 'A_CONV/adapt2_11_1.12883789e-03/phi_000000000000.h5', savepng=True, cmap='rainbow', caxis=[0,1] )
#wabbit_tools.plot_wabbit_file( 'A_CONV/adapt2_11_1.12883789e-03/phi_000000500000.h5', savepng=True, cmap='rainbow', caxis=[0,1] )
#wabbit_tools.plot_wabbit_file( 'A_CONV/adapt2_11_1.12883789e-03/phi_000001000000.h5', savepng=True, cmap='rainbow', caxis=[0,1] )
#
#wabbit_tools.plot_wabbit_file( 'B_SWIRL/dt1_equi_4th-4th-4th_1.0e-3/phi_000000000000.h5', savepng=True, cmap='rainbow', caxis=[0,1] )
#wabbit_tools.plot_wabbit_file( 'B_SWIRL/dt5_nonequi_4th-4th-4th_2.5e-3/phi_000000000000.h5', savepng=True, cmap='rainbow', caxis=[0,1] )

#
#
#files = glob.glob('B_SWIRL/dt5_nonequi_4th-4th-4th_1.0e-3/phi_*.h5', recursive=True)
#files = glob.glob('./B_SWIRL/adaptive1*/phi_000005000000.h5', recursive=True)
#files.extend(glob.glob('./B_SWIRL/adaptive1*/phi_000002500000.h5', recursive=True))
#files.extend(glob.glob('./B_SWIRL/adaptive1*/phi_000010000000.h5', recursive=True))
#files = glob.glob('../swirl/meso/swirl_1*/phi_*.h5', recursive=True)

#r = "/home/engels/dev/WABBIT-convergence-test/C_HALFSWIRL_newcode/adaptive_halfswirl_Bs33_i21_JmaxInf_eps1.00000000e-06"
#files=[r+'/phi_000000000000.h5', r+'/phi_000002500000.h5']
#
#plt.close('all')
#for file, i in zip(files, range(len(files))):
#
#    wabbit_tools.plot_wabbit_file( file, savepng=True, cmap='rainbow', ticks=False,
#                                  colorbar=True, caxis=[0,1], dpi=600, title=False, block_edge_color='w' )
#    wabbit_tools.plot_wabbit_file( file, savepng=True, cmap='rainbow', ticks=False,
#                                  colorbar=False, caxis=[0,1], dpi=600, gridonly=True, title=False, block_edge_color='w' )
##    print(f2)

#    if not os.path.isfile( f2 ):
#        wabbit_tools.plot_wabbit_file( file, savepng=True, cmap='rainbow', caxis=[0,1] )
#        wabbit_tools.plot_wabbit_file( file, savepng=True, gridonly=True )
##    else:
#        print('already taken care of')
#    print("[%i %i]" % (i, len(files)) )



#wabbit_tools.plot_wabbit_file( '/home/engels/dev/WABBIT4-new-physics/phi_000000000000.h5', savepng=True, cmap='rainbow', ticks=False,
#                                  colorbar=True, dpi=600, gridonly=True, title=False, block_edge_color='w' )
#
#plt.figure()
#wabbit_tools.plot_wabbit_file( '/home/engels/dev/WABBIT4-new-physics/phi_000000100000.h5', savepng=True, cmap='rainbow', ticks=False,
#                                  colorbar=True, dpi=600, gridonly=True, title=False, block_edge_color='w' )
import farge_colormaps

#plt.close('all')
fig = plt.figure(2)
a = 2
plt.gcf().set_size_inches( [5.0*a, 5.0*a] )
plt.gcf().subplots_adjust(top=0.95, bottom=0.05,left=0.02, right=0.97)

overwrite = True

# for prefix, cmap in zip(['vort', 'p', 'div', 'Ux', 'Uy'],['vorticity','pressure','streamfunction','velocity','velocity']):
for prefix, cmap in zip(['div'],['streamfunction']):
    files = glob.glob('/home/engels/dev/WABBIT4-new-physics/WABBIT_div_test/'+prefix+'_*.h5')
    files.sort()

    fc = farge_colormaps.farge_colormap_multi( taille=600, limite_faible_fort=0.2, etalement_du_zero=0.02, type=cmap )

    for file, i in zip(files[1:], range(len(files[1:]))):
        file2 = file.replace('.h5','.png')
        if not os.path.isfile( file2 ) or overwrite:
            wabbit_tools.plot_wabbit_file(file, savepng=True, cmap=fc, ticks=False, colorbar=True, dpi=300, caxis=[-0.06, 0.06],
                                          title=True, block_edge_color='w', caxis_symmetric=True, shading='gouraud' )
        else:
            print("File "+file2+" exists.")

#files = glob.glob('/home/engels/dev/WABBIT4-new-physics/WABBIT_div_test/'+'vort'+'_*.h5')
#files.sort()
#fc = farge_colormaps.farge_colormap_multi( taille=256, limite_faible_fort=0.2, etalement_du_zero=600.0*3./256., type='vorticity' )
#wabbit_tools.plot_wabbit_file( files[-1], savepng=True, cmap=fc, ticks=False, colorbar=True, dpi=300, title=True, block_edge_color='w',caxis_symmetric=True, shading='gouraud' )