#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
##############################################
WAVELET PROPPER ORTHOGONAL DECOMPOSITION

MAIN skript calling all subroutines vortex street
##############################################
"""
from wPOD.run_wabbit import run_wabbit_POD, compute_vorticity
import numpy as np
from wPOD import adapt2eps
from wPODdirs import *
from wPOD.wPODerror import *
from wPOD.sci_notation import sci_notation
from wPOD import plot_vector_modes
from matplotlib.offsetbox import AnchoredText
import os
import farge_colormaps
from matplotlib import rc
font = {'family' : 'serif',
        'size'   : 22}
rc('font',**font)
rc('text', usetex=True) 
###############################################################################
###############################################################################
# directories needed
dir_list={}
dir_list["work"]=dirs["work"] +"/cyl/"
dir_list["images"]= dirs["images"] + "/cyl/"
dir_list["wabbit"]= wdir

data = {"folder" :  home+"/develop/results/cyl_new/ACM_cylinder_equi/", # where the data is stored
        "qname" : ["ux", "uy", "p"]
        }

###############################################################################
###############################################################################
# %% In this step we call all necessary wabbit routines
#           * --POD
#           * --POD-reconstruct
#           * --PODerror 
#  for the given parameters
###############################################################################
""" 
 Vortex Street
"""
Jmax_list = [6]
Jmax_dir_list = [ "Jmax"+str(Jmax)+"/" for Jmax in Jmax_list]
eps_list = np.asarray([0]+[float("%1.1e" %eps) for eps in np.logspace(-5,0,10)])
eps_list = np.delete(eps_list,2)
#eps_list = np.asarray([0]+[float("%1.1e" %eps) for eps in np.logspace(-7,0,10)])
eps_dir_list = [ "eps%1.1e" %eps for eps in eps_list]


mode_lists = ["mode1_list.txt","mode2_list.txt","mode3_list.txt","mode4_list.txt"]
reconstructed_iteration =7

# %%
#run_wabbit_POD(wabbit_setup, dir_list, data, Jmax_list, eps_list, mode_lists, reconstructed_iteration)



###############################################################################
###############################################################################
# %% 

delta_err,clist=plot_wPODerror(Jmax_list, Jmax_dir_list,eps_dir_list,eps_list, dir_list, show_legend=True)
# %% Compression ERROR
wdir = dirs["wabbit"]
fig_adapt = [plt.figure(41)]
ax_adapt = [fig.add_subplot() for fig in fig_adapt]
h5_fname = [sorted(glob.glob(data["folder"]+'/Jmax6/'+qty+'*.h5'))[100] for qty in data["qname"]]
[l2error, linferror, Nblocks, Nblocksdense]=adapt2eps(h5_fname, \
wdir, eps_list, wabbit_setup['memory'], wabbit_setup['mpicommand'], wavelets="CDF40",\
normalization = "L2",create_log_file=True, show_plot=True)

# %% get the data
# COLORMAP:
fc = farge_colormaps.farge_colormap_multi( taille=600,limite_faible_fort=0.2, etalement_du_zero=0.04, type='vorticity' )
plt.close("all")

fig, axes = plt.subplots(nrows=3, ncols=2)
fig.subplots_adjust(bottom=0.001,top = 0.999,wspace=0.01, hspace=-0.8)

for i, eps in enumerate(eps_list[::2]):
    
    files=sorted(glob.glob('./adapt_files/u[x,y]-eps_%1.1e_'%eps+'*.h5'))
    success=compute_vorticity(wdir, files[0], files[1], wabbit_setup["memory"], "")
    file = files[0].replace('ux-eps_%1.1e_'%eps,'vorx_')
    file = os.path.split(file)[1]
    ax,cb,hplot = wt.plot_wabbit_file(file,cmap=fc,caxis_symmetric=True,dpi=300, \
                                 shading='flat', caxis=[-6,6], \
                                 block_edge_alpha=0.1,colorbar_orientation="horizontal", \
                                 title=False, ticks=False, colorbar=False, fig=fig, ax=axes.flat[i])
    at = AnchoredText("$\\epsilon= "+sci_notation(eps,1)+'$',
                  prop=dict(size=9), frameon=True,
                  loc='upper right',
                  )
    at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
    axes.flat[i].add_artist(at)
    #cb.set_label("vorticity [1/s]")
#    print("Saving plot to: "+pic_dir+plt_file)
    #plt.savefig( pic_dir+plt_file, dpi=300, transparent=True, bbox_inches='tight' )
    plt.show()

#fig.subplots_adjust(bottom=0.001)
cbar_ax = fig.add_axes([0.335, 0.26, 0.35, 0.01])
cbar=fig.colorbar(hplot, cax=cbar_ax, orientation="horizontal")
cbar.ax.tick_params(labelsize=11)
cbar=fig.colorbar(hplot, cax=cbar_ax, orientation="horizontal")
plt.savefig( dir_list["images"]+ 'vorx_adapt.png', dpi=600, transparent=True, bbox_inches='tight' )
##############################################################################

# %% plot modes
# 
import os
vorx_files = []
p_files = []
for i, eps in enumerate(eps_list[:1]):
    for nmode in range(30):
        eps=1.3e-04
        files = [sorted(glob.glob(dir_list["work"]+"Jmax6/eps%1.1e"%eps+'/'+qty+'*.h5'))[nmode] for qty in ["mode1","mode2","mode3"]]
        success=compute_vorticity(wdir, files[0], files[1], wabbit_setup["memory"], "")
        file = files[0].replace('mode1_','vorx-mode_')
        file_0 = "vorx_000000000000.h5"
        os.system("mv "+file_0+" "+file)
        #print(files)
        vorx_files.append(file)
        p_files.append(files[2])

quantity='mode3'

modes_dict={0:vorx_files,1:p_files}
acoef_file = dir_list["work"]+"Jmax%d"%Jmax_list[-1]+"/eps%1.1e"%eps+"/a_coefs.txt"    
plot_vector_modes(modes_dict, acoef_file, quantity,"mode-eps%1.1e"%eps,pic_dir=dir_list["images"],mode_names=["$\\nabla\\times \\mathbf{v}$",'$p$'])

# %% Plot reconstructed snapshots
    
import matplotlib.gridspec as gs
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

fc = farge_colormaps.farge_colormap_multi( taille=600,limite_faible_fort=0.2, etalement_du_zero=0.04, type='vorticity' )

vorx_files_sparse = []
p_files_sparse = []
eps =1.3e-04

for nmode in range(29):
        files = [sorted(glob.glob(dir_list["work"]+"Jmax6/eps%1.1e"%eps+'/'+qty+'*.h5'))[nmode] for qty in ["reconst1","reconst2","reconst3"]]
        file_ux = os.path.relpath(files[0],'.')
        file_uy = os.path.relpath(files[1],'.')
        success=compute_vorticity(wdir, file_ux, file_uy, wabbit_setup["memory"], "")
        file = files[0].replace('reconst1-','vorx-recon-')
        file_0 = "vorx_000100000000.h5"
        os.system("mv "+file_0+" "+file)
        vorx_files_sparse.append(file)
        p_files_sparse.append(files[2])

# %%
eps =0 
vorx_files_dense = []
p_files_dense = []
for nmode in range(29):
        files = [sorted(glob.glob(dir_list["work"]+"Jmax6/eps%1.1e"%eps+'/'+qty+'*.h5'))[nmode] for qty in ["reconst1","reconst2","reconst3"]]
        file_ux = os.path.relpath(files[0],'.')
        file_uy = os.path.relpath(files[1],'.')
        success=compute_vorticity(wdir, file_ux, file_uy, wabbit_setup["memory"], "")
        file = files[0].replace('reconst1-','vorx-recon-')
        file_0 = "vorx_000*00.h5"
        os.system("mv "+file_0+" "+file)
        #print(files) 
        vorx_files_dense.append(file)
        p_files_dense.append(files[2])
# %%
import re
fig, axes = plt.subplots(nrows=3, ncols=2)
fig.subplots_adjust(bottom=0.001,top = 0.999,wspace=0.01, hspace=-0.8)


for i,fsparse in enumerate(vorx_files_sparse[1:11:4]):
    ax,cb,hplot = wt.plot_wabbit_file(fsparse,cmap=fc, caxis_symmetric=True,\
                                block_edge_alpha=0.1,shading='flat',caxis=[-6,6],\
                                ticks=False ,title=False, colorbar=False, fig=fig, ax=axes[i,0])
    #match = re.search('vorx-recon-(\d+)', fsparse)
    match = re.search('vorx-recon-(\d+)', fsparse)
    r = int(match.group(1))
    at = AnchoredText("$r= "+sci_notation(r,1)+"$",
                  prop=dict(size=11), frameon=True,
                  loc='upper right',
                  )
    at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
    axes[i,0].add_artist(at)
    #cb.set_label("vorticity [1/s]")
    plt.show()

for i,fdense in enumerate(vorx_files_dense[1:11:4]):
    ax,cb,hplot = wt.plot_wabbit_file(fdense,cmap=fc, caxis_symmetric=True,\
                                block_edge_alpha=0.1,shading='flat',caxis=[-6,6],\
                                ticks=False ,title=False, colorbar=False, fig=fig, ax=axes[i,1])
    #match = re.search('vorx-recon-(\d+)', fsparse)
    match = re.search('vorx-recon-(\d+)', fdense)
    r = int(match.group(1))
    at = AnchoredText("$r=$"+sci_notation(r,1),
                  prop=dict(size=10), frameon=True,
                  loc='upper right',
                  )
    at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
    axes[i,1].add_artist(at)
    #cb.set_label("vorticity [1/s]")
    plt.show()

cbar_ax = fig.add_axes([0.335, 0.26, 0.35, 0.01])
cbar=fig.colorbar(hplot, cax=cbar_ax, orientation="horizontal")
cbar.ax.tick_params(labelsize=9)
cbar=fig.colorbar(hplot, cax=cbar_ax, orientation="horizontal")
plt.savefig( dir_list["images"]+ 'vorx-recon.png', dpi=600, transparent=True, bbox_inches='tight' )   