# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import os
import farge_colormaps
import glob
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc
import wabbit_tools as wt
import glob
import pathlib, sys
import matplotlib as mpl
from tabulate import tabulate
mpl.style.use('default')
file_path = pathlib.Path(__file__).parent.absolute()
sys.path.append(str(file_path) + "/../LIB")
from plot_utils import save_fig
from utils import read_performance_file
from numpy import pi
###############################################################################
# LATEX FONT
font = {'family' : 'serif',
        'size'   : 18}
rc('font',**font)
rc('text', usetex=True)   
###############################################################################
# COLORMAP:
fc = farge_colormaps.farge_colormap_multi( taille=600,limite_faible_fort=0.2, etalement_du_zero=0.04, type='pressure' )
###############################################################################
# %% Specify:
###############################################################################
# %% get the data
quantity="q_"
plt.close("all")
case = "CDF44_disc"
ini = "disc"+"-convection.ini"
eps_dir="./adaptive_"+ case +"_Bs17_Jmax*/"
norm = 2
# %%

#files = glob.glob(eps_dir+'/'+quantity+'*000001000000.h5')
files = glob.glob("stability_CDF44_disc_Bs17_Jmax7_eps1.0e-07/"+'/'+quantity+'*.h5')
files.sort()
for file in files:
    plt_file = file.replace('h5','png')
    fig, ax = plt.subplots() 
    ax,cb,hplot = wt.plot_wabbit_file(file,cmap ='bwr', dpi=300, \
                                shading='gouraud', caxis=[0,1],block_edge_color='w', \
                                block_edge_alpha=0.7, block_linewidth=0.9 ,colorbar_orientation="vertical", \
                                title=False, ticks=False, colorbar=True)
    #ax.set_title("$\epsilon="+str(eps_list[i])+"$")
   # ax.set_xlabel("$x$")
    #ax.set_ylabel("$y$")
    ax.set_aspect('equal')

    #cb.set_label("vorticity [1/s]")
    print("Saving plot to: "+plt_file)
    plt.savefig( plt_file, dpi=300, transparent=True, bbox_inches='tight' )
    plt.show()
    #tikzplotlib.clean_figure()
    #tikzplotlib.save(plt_file)
    
##############################################################################
# %% eps vs err
##############################################################################

def make_equidistant(file, level, wdir = "./", mpi = "mpirun -np 4 ", predictor_order = "4"):
    
    file_dense = file.replace("_","-dense_")
    if (os.path.isfile(file_dense)):
        return 0
    
    command = mpi + wdir + "wabbit-post --sparse-to-dense " + file +" "+ file_dense +" " + str(level) +" " + predictor_order
    success = os.system(command)
    
    if success != 0:
        print("command did not execute successfully")
        
    return success

def chdir_make_equidistant(directory, *args, **kwargs):
    cwd = os.getcwd()
    os.chdir(directory)
    make_equidistant(*args, **kwargs)
    
    os.chdir(cwd)


def adaptive(rootdir, norm, ref, file,inifile="blob-convection.ini"):
    dirsx = glob.glob( rootdir )
    e, Nb, eps_list, Jmax_list, compression_list = [], [], [], [], []
    if len(dirsx)==0:
        raise ValueError('no data')
    
    for d in dirsx:
        eps = wt.get_ini_parameter(  d+"/"+inifile, "Blocks", "eps")
        Jmax = wt.get_ini_parameter(  d+"/"+inifile, "Blocks", "max_treelevel", dtype=np.int32)
        eps_list.append(eps)
        Jmax_list.append(Jmax)
        
        fpath = d+"/"+file
        fpath_sparse = d+"/"+file.replace("-dense","")
        if (os.path.isfile(fpath)):
            # compute error
            err = wt.wabbit_error_vs_wabbit( fpath, ref, norm=norm )
            e.append( err )
            if (os.path.isfile(fpath_sparse)):
                Nbl_sparse = sum(wt.block_level_distribution_file( fpath_sparse))
                Nbl_dense = sum(wt.block_level_distribution_file( fpath))
                compression_list.append(Nbl_sparse/Nbl_dense) 
                

    return e,Jmax_list,eps_list, compression_list

def err_equidist(rootdir, norm, ref, file, inifile="blob-convection.ini"):
    dirsx = glob.glob( rootdir )
    e, Jmax_list = [], []
    if len(dirsx)==0:
        raise ValueError('no data')
    
    for d in dirsx:
        Jmax = wt.get_ini_parameter(  d+"/"+inifile, "Blocks", "max_treelevel", dtype=np.int32)
        Jmax_list.append(Jmax)
        
        fpath = d+"/"+file
        fpath_ref = d+"/"+ref
        
        if (os.path.isfile(fpath)):
            # compute error
            err = wt.wabbit_error_vs_wabbit( fpath, fpath_ref, norm=norm )
            e.append( err )
                

    return e,Jmax_list



def disc_analytic2wabbit(time, Jmax, L=[1,1], Bs=[17,17], dir = "./"):
    
    def init(x,L):
       R = np.asarray(L[0])*0.15
       lambd = np.asarray(L[0])*0.001
       x0    = [L[0]/2 + L[0]/4*np.cos(2*pi*time), L[1]/2 + L[1]/4*np.sin(2*pi*time)]
       xrel  = [x-x0 for x,x0 in zip(x,x0)]
       field = np.ones_like(x[0])
       field =  1/(1+np.exp( (np.sqrt(xrel[0]**2 + xrel[1]**2) - R ) /lambd))
       return field
   
    dim = len(L)
    N = [2**Jmax*(Bs[d]-1) for d in range(dim)]
    x = [np.linspace(0,L[d],N[d]) \
         for d in range(dim) ]
    dx = [x[d][1]-x[d][0] for d in range(dim)]
    print("dx = ",min(dx))
    print("Ngrid = ", N)
    X = np.meshgrid(*x)
    phi = init(X,L)
    
    # create reference file
    file_ref = wt.dense_to_wabbit_hdf5(phi,dir+"/q-ref", Bs, L,1)
    return file_ref



# %%
field, box, dx, X = wt.to_dense_grid("./adaptive_"+case+"_Bs17_Jmax7_eps1.0e-10/q-dense_000001000000.h5")
Xgrid = np.meshgrid(*X)
L = [1,1]
time = 0
R = np.asarray(L[0])*0.15
lambd = np.asarray(L[0])*0.005
x0    = [L[0]/2 + L[0]/4*np.cos(2*pi*time), L[1]/2 + L[1]/4*np.sin(2*pi*time)]
field_ref = np.sqrt((Xgrid[0]-x0[0])**2 + (Xgrid[1]-x0[1])**2)-R
field_ref =  1/(1+np.exp( field_ref /lambd))
#field_ref=  0.5*(1-np.tanh( field_ref /lambd))


plt.pcolormesh(field_ref-field)
plt.colorbar()
plt.show()
print(np.linalg.norm(field-field_ref)/np.linalg.norm(field_ref))
# %% 

rootdir = "./adap*"+case+"*"
dirsx = glob.glob( rootdir )
for d in dirsx:
    chdir_make_equidistant(d,"q_000000000000.h5",7,wdir="../../../")
    chdir_make_equidistant(d,"q_000001000000.h5",7,wdir="../../../")


error_list,Jmax_list,eps_list, compress_list = adaptive("./ada*"+case+"*Jmax*",norm,
                                         #ref=file_ref,
                                         #ref = "adaptive_"+case+"_Bs17_Jmax5_eps1.0e-10/q-dense_000000000000.h5", 
                                         ref = "disc_ref_Jmax7/q_000000000000.h5", 
                                         file="q-dense_000001000000.h5",
                                         inifile=ini)

rootdir = "./equid*disc"+"*Jmax*"
error_equi,Jmax_equi = err_equidist("./equid*disc"+"*Jmax*",norm,
                                         #ref=file_ref,
                                         #ref = "adaptive_"+case+"_Bs17_Jmax5_eps1.0e-10/q-dense_000000000000.h5", 
                                         ref = "q_000000000000.h5", 
                                         file="q_000001000000.h5",
                                         inifile=ini)


perf_files = glob.glob( rootdir+"/"+"performan*" )
dat_list_equi = read_performance_file( perf_files)
tcpu_list_equi = [sum(dat[:,2]*dat[:,-1]) for dat in dat_list_equi]

Jmax= np.unique(Jmax_list)
eps = np.flip(np.unique(eps_list))
errors_by_lvl =dict((el,[]) for el in Jmax)
eps_by_lvl =dict((el,[]) for el in Jmax)
compress_by_lvl =dict((el,[]) for el in Jmax)
for k, (lvl,epsilon,err) in enumerate(zip(Jmax_list,eps_list,error_list)):
    eps_by_lvl[lvl].append(epsilon)
    errors_by_lvl[lvl].append(err)
    compress_by_lvl[lvl].append(2**7*compress_list[k]*16**2)

# %% ERROR vs EPS
from matplotlib.lines import Line2D
from matplotlib import lines
import matplotlib as mpl
lines =['-', '--', '-.', ':']
markers = list(Line2D.markers.keys())

fig,ax = plt.subplots()
ax.set_yscale('log')
ax.set_xscale('log')
plt.minorticks_on()
for j,level in enumerate(Jmax):
    if level>1:
        p=ax.loglog(eps_by_lvl[level],errors_by_lvl[level],lines[j%4] + markers[j], label=r"$ J_\mathrm{max}=%d $"%level)
        ax.hlines(error_equi[j],np.min(eps),np.max(eps),linestyle=':',color=p[0].get_color(),alpha=1)
lin= wt.logfit(eps_by_lvl[Jmax[-1]][1:8],errors_by_lvl[Jmax[-1]][1:8])
ax.loglog(eps, 10**(lin[1]) * eps**lin[0],'k--', label=r"$%1.1f\epsilon^{%1.1f}$"%(10**lin[1],lin[0]))
ax.set_xlabel(r"threshold $\epsilon$"  )
locmin = matplotlib.ticker.LogLocator(base=10.0,subs=(0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9),numticks=12)
ax.xaxis.set_minor_locator(locmin)
ax.yaxis.set_minor_locator(locmin)
ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
plt.ylabel(r"$\Vert q-q^{\epsilon,\Delta t,h}\Vert_%d/\Vert q\Vert_%d$"%(norm,norm))
plt.legend(fontsize="small")
plt.grid(which="both",linestyle=':')
save_fig(case+"_errors_vs_eps",strict=True)

eps_opt = np.asarray([10**(1/lin[0]*np.log10(err/10**(lin[1]))) for err in error_equi])
print(eps_opt)
for epsilon in eps_opt:
    plt.vlines(epsilon,np.min(eps),np.max(eps),linestyle="--",color='k')
# plt.xlim([0.5e-4,1])
# plt.ylim([0.5e-5,1])
# %%

rootdir = "./opt_eps*"+case+"*"
dirsx = glob.glob( rootdir )
for d in dirsx:
    chdir_make_equidistant(d,"q_000000000000.h5",7,wdir="../../../")
    chdir_make_equidistant(d,"q_000001000000.h5",7,wdir="../../../")


perf_files = glob.glob( rootdir+"/"+"performan*" )
dat_list = read_performance_file( perf_files)
tcpu_list = [sum(dat[:,2]*dat[:,-1]) for dat in dat_list]

error_list,Jmax_list,eps_list, compress_list = adaptive("./opt_eps*"+case+"*Jmax*",norm,
                                         #ref=file_ref,
                                         #ref = "adaptive_"+case+"_Bs17_Jmax5_eps1.0e-10/q-dense_000000000000.h5", 
                                         ref = "disc_ref_Jmax7/q_000000000000.h5", 
                                         file="q-dense_000001000000.h5",
                                         inifile=ini)



Jmax= np.unique(Jmax_list)
eps = np.flip(np.unique(eps_list))
errors_by_lvl =dict((el,[]) for el in Jmax)
eps_by_lvl =dict((el,[]) for el in Jmax)
DOFs_by_lvl =dict((el,[]) for el in Jmax)
for k, (lvl,epsilon,err) in enumerate(zip(Jmax_list,eps_list,error_list)):
    eps_by_lvl[lvl].append(epsilon)
    errors_by_lvl[lvl].append(err)
    DOFs_by_lvl[lvl].append((2**7*compress_list[k]*2**7)*16**2)


# %% ERROR vs DOFS
from matplotlib.lines import Line2D
from matplotlib import lines
import matplotlib as mpl
lines =['-', '--', '-.', ':']
markers = list(Line2D.markers.keys())

fig,ax = plt.subplots()
for j,level in enumerate(Jmax):
    if level>1:
        ax.semilogy(DOFs_by_lvl[level]/(2**level*2**level)/16**2,errors_by_lvl[level],'o', label=r"$ J_\mathrm{max}=%d $"%level)
    
#ax.loglog(eps,eps, 'k--', label = r"$\epsilon$")
ax.set_xlabel(r"compression factor"  )
plt.ylabel(r"$\Vert q-q^{\epsilon,\Delta t,h}\Vert_%d/\Vert q\Vert_%d$"%(norm,norm))
plt.legend(fontsize="small")
plt.grid(which="both",linestyle=':')
save_fig(case+"_errors_vs_DOFS",strict=True)
# plt.xlim([0.5e-4,1])
# plt.ylim([0.5e-5,1])
DOFs_equi = [(2**lvl*16)**2 for lvl in Jmax]

#tcpu_list_equi[-1] = np.size(dat_list[-1],0)*tcpu_list_equi[-1]/np.size(dat_list_equi[-1],0)
table = np.stack((Jmax,np.concatenate(list(DOFs_by_lvl.values())),np.concatenate(list(errors_by_lvl.values())),tcpu_list,DOFs_equi,error_equi,tcpu_list_equi),axis=1)
print(tabulate(table,headers=["max. level","DOFs (adapt)", "rel. error","cpu-time (s)","DOFs (equi)", "rel. error","cpu-time (s)"], tablefmt='latex_booktabs',floatfmt=(".0f", ".0f", ".1e", ".0f",".0f", ".1e",".0f")))

# %% spatial convergence


dx = np.asarray([1/(2**(j)*16) for j in Jmax])

errors = [errors_by_lvl[lvl][-1] for lvl in Jmax]
lin = wt.logfit(dx, errors)
plt.loglog(dx, errors,'-o',label="adaptive $\epsilon^\mathrm{opt}(J_\mathrm{max})$")
wt.add_convergence_labels(dx, errors)
plt.loglog(dx,error_equi,'--',label="equidistant")
#plt.loglog(dx, 10**(lin[1]) * dx**lin[0],'k--', label=r"$\mathcal{O}(\Delta x^{%1.1f}$)"%lin[0])
plt.xlabel(r"lattice spacing $h$")
plt.grid(which="both",linestyle=':')
plt.ylabel(r"$\Vert q-q^{\epsilon,\Delta t,h}\Vert_%d/\Vert q\Vert_%d$"%(norm,norm))
plt.legend()
save_fig(case+"_errors_vs_lattice_spacing",strict=True)

# %% 
norm = 2
file_ref = disc_analytic2wabbit(0, 6)
error_list,Jmax_list,eps_list, compression_list = adaptive("./ada*"+case+"*Jmax*",norm,
                                         #ref=file_ref,
                                         #ref = "adaptive_"+case+"_Bs17_Jmax5_eps1.0e-10/q-dense_000000000000.h5", 
                                         ref = "disc_ref_Jmax7/q_000000000000.h5", 
                                         file="q-dense_000000000000.h5",
                                         inifile=ini)
Jmax= np.unique(Jmax_list)
eps = np.flip(np.unique(eps_list))
errors_by_lvl =dict((el,[]) for el in Jmax)
eps_by_lvl =dict((el,[]) for el in Jmax)
for k, (lvl,epsilon,err) in enumerate(zip(Jmax_list,eps_list,error_list)):
    eps_by_lvl[lvl].append(epsilon)
    errors_by_lvl[lvl].append(err)

# %%
from matplotlib.lines import Line2D
from matplotlib import lines
lines =['-', '--', '-.', ':']
markers = list(Line2D.markers.keys())

for j,level in enumerate(Jmax):
    if level>1:
        plt.loglog(eps_by_lvl[level],errors_by_lvl[level],lines[j%4] + markers[j], label=r"$ J_\mathrm{max}=%d $"%level)
    
plt.loglog(eps,eps, 'k--', label = r"$\epsilon$")
plt.xlabel(r"$\epsilon$"  )
plt.ylabel(r"$\Vert q-q^{\epsilon}\Vert_%d/\Vert q\Vert_%d$"%(norm,norm))
plt.legend()
save_fig(case+"_compression_errors_vs_eps",strict=True)
# plt.xlim([0.5e-4,1])
# plt.ylim([0.5e-5,1])

# %%
perf_dat = np.loadtxt("adaptive_CDF44_disc_Bs17_Jmax7_eps1.0e-05/performance.t")
#perf_dat = np.loadtxt("stability_CDF44_disc_Bs17_Jmax7_eps1.0e-07/performance.t")

plt.plot(perf_dat[::10,0],perf_dat[::10,4])
#perf_dat = np.loadtxt("adaptive_CDF44_disc_Bs17_Jmax7_eps1.0e-05/performance.t")
plt.xlabel(r"time $t$")
plt.ylabel("Number of Blocks")
plt.xlim([0,1])
save_fig(case+"_Nblocks_vs_time",strict=True)