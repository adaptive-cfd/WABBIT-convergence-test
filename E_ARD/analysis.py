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
from numpy.linalg import norm
from tabulate import tabulate
mpl.style.use('default')
file_path = pathlib.Path(__file__).parent.absolute()
sys.path.append(str(file_path) + "/../LIB")
from plot_utils import save_fig
from fourier_solver import solve_pacman
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
case = "CDF44_dealias_pacman"
#case = "CDF44_pacman"
ini = "pacman.ini"
eps_dir="./adaptive_"+ case +"_Bs17_Jmax6_eps2.8*/"
norm = 2
Bs = 17
# %%

files = glob.glob(eps_dir+'/'+quantity+'*.h5')
files = glob.glob('opt_eps*Jmax6*/*.h5')
files.sort()

for file in files:
    plt_file = file.replace('h5','png')
    if (os.path.isfile(plt_file)):
        continue
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
            field, box, dx, X = wt.to_dense_grid(fpath)
            err = np.linalg.norm(field-ref, ord=norm)/np.linalg.norm(ref, ord = norm)
            e.append( err )
            print("rel error: %f" % err)
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
        
        if (os.path.isfile(fpath)):
            # compute error
            field, box, dx, X = wt.to_dense_grid(fpath)
            err = np.linalg.norm(field-ref, ord=norm)/np.linalg.norm(ref, ord = norm)
            e.append( err )
            print("rel error: %f" % err)
                

    return e,Jmax_list

def evaluate_performance(rootdir,field_ref, ini="pacman.ini", h5_file_name="q_000001000000.h5", norm=2):
    
    dirsx = glob.glob( rootdir )
    for d in dirsx:
        chdir_make_equidistant(d,"q_000001000000.h5",7,wdir="../../../")
    
    
    perf_files = list(np.sort(glob.glob( rootdir+"/"+"performan*" )))
    dat_list = read_performance_file( perf_files)
    tcpu_list = [sum(dat[:,2]*dat[:,-1]) for dat in dat_list]
    DOFs = [np.mean(dat[:,3])*(Bs-1)**2 for dat in dat_list]
    
    file_dense = h5_file_name.replace("q", "q-dense")
    error_list,Jmax_list,eps_list, compress_list = adaptive(rootdir+"*Jmax*",norm,
                                             #ref=file_ref,
                                             #ref = "adaptive_"+case+"_Bs17_Jmax5_eps1.0e-10/q-dense_000000000000.h5", 
                                             ref = field_ref, 
                                             file=file_dense,
                                             inifile=ini)
    ind = np.argsort(Jmax_list)
    error_list = [error_list[i] for i in ind]
    eps_list = [eps_list[i] for i in ind]
    compress_list = [compress_list[i] for i in ind]
    Jmax_list = [Jmax_list[i] for i in ind]

    return {"Jmax": Jmax_list,"error":error_list, "eps":eps_list, "DOF": DOFs, "tcpu":tcpu_list, "performance_files":perf_files}


# %% 

rootdir = "./adap*"+case+"*"
dirsx = glob.glob( rootdir )

for d in dirsx:
    chdir_make_equidistant(d,"q_000000000000.h5",7,wdir="../../../")
    chdir_make_equidistant(d,"q_000001000000.h5",7,wdir="../../../")
    

rootdir = "./equi*pacman*"
dirsx = glob.glob( rootdir )
for d in dirsx:
    chdir_make_equidistant(d,"q_000000000000.h5",7,wdir="../../../")
    chdir_make_equidistant(d,"q_000001000000.h5",7,wdir="../../../")

# %%    
    
field, box, dx, X = wt.to_dense_grid("./adaptive_CDF44_dealias_pacman_Bs17_Jmax7_eps1.0e-04/q-dense_000001000000.h5")
# %%
Xgrid = np.meshgrid(*X)

field_ref = np.sqrt((Xgrid[0]-0.4)**2 + (Xgrid[1]-0.5)**2)-0.2
field_ref=  0.5*(1-np.tanh( field_ref /0.005))


plt.pcolormesh(field_ref-field)
plt.colorbar()
plt.show()
print(np.linalg.norm(field-field_ref)/np.linalg.norm(field_ref))
# %%
#q_ref = solve_pacman(Ngrid=np.shape(field))
q_ref = np.load("fft_pacman_ref_2048.npy")
#tcpu_fft=6775.762
field_ref = q_ref[...,-1]
plt.pcolormesh(abs(field_ref-field))
plt.clim(vmin=0,vmax=0.8e-4)
plt.colorbar()
print(np.linalg.norm(field-field_ref)/np.linalg.norm(field_ref))
plt.savefig("error_eps1e-4_iter1.png")
plt.show()
#%%
error_list,Jmax_list,eps_list, compress_list = adaptive("./ada*"+case+"*Jmax*",norm,
                                         #ref=file_ref,
                                         #ref = "adaptive_"+case+"_Bs17_Jmax5_eps1.0e-10/q-dense_000000000000.h5", 
                                         ref = field_ref, 
                                         file="q-dense_000001000000.h5",
                                         inifile=ini)
#%%
rootdir = "./equi_*deal*pacman"+"*Jmax*"
#rootdir = "./equi_pacman"+"*Jmax*"
error_equi,Jmax_equi = err_equidist(rootdir,norm,
                                          #ref=file_ref,
                                          #ref = "adaptive_"+case+"_Bs17_Jmax5_eps1.0e-10/q-dense_000000000000.h5", 
                                          ref = field_ref, 
                                          file="q-dense_000001000000.h5",
                                          inifile=ini)
ind = np.argsort(Jmax_equi)
error_equi = [error_equi[i] for i in ind]
Jmax_equi = [Jmax_equi[i] for i in ind]


perf_files_equi = list(np.sort(glob.glob( rootdir+"/"+"performan*" )))
dat_list_equi = read_performance_file( perf_files_equi)
tcpu_list_equi = [sum(dat[:,2]*dat[:,-1]) for dat in dat_list_equi]
DOFs_equi = [np.max(dat[:,3])*(Bs-1)**2 for dat in dat_list_equi]

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
    if level<7:
        p=ax.loglog(eps_by_lvl[level],errors_by_lvl[level],lines[j%4] + markers[j], label=r"$ J_\mathrm{max}=%d $"%level)
        ax.hlines(error_equi[j],np.min(eps),np.max(eps),linestyle=':',color=p[0].get_color(),alpha=1)
lin= wt.logfit(eps_by_lvl[Jmax[-3]][1:5],errors_by_lvl[Jmax[-3]][1:5])
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
# print(eps_opt)
# for epsilon in eps_opt:
#     plt.vlines(epsilon,np.min(eps),np.max(eps),linestyle="--",color='k')

# %%

rootdir = "./opt_eps*"+case+"*"
dirsx = glob.glob( rootdir )
for d in dirsx:
    chdir_make_equidistant(d,"q_000000000000.h5",7,wdir="../../../")
    chdir_make_equidistant(d,"q_000001000000.h5",7,wdir="../../../")


perf_files_opt = list(np.sort(glob.glob( rootdir+"/"+"performan*" )))
dat_list = read_performance_file( perf_files_opt)
tcpu_list_opt = [sum(dat[:,2]*dat[:,-1]) for dat in dat_list]
DOFs_opt = [np.mean(dat[:,3])*(Bs-1)**2 for dat in dat_list]

error_list_opt,Jmax_list_opt,eps_list_opt, compress_list_opt = adaptive("./opt_eps*"+case+"*Jmax*",norm,
                                         #ref=file_ref,
                                         #ref = "adaptive_"+case+"_Bs17_Jmax5_eps1.0e-10/q-dense_000000000000.h5", 
                                         ref = field_ref, 
                                         file="q-dense_000001000000.h5",
                                         inifile=ini)



# %% ERROR vs DOFS
from matplotlib.lines import Line2D
from matplotlib import lines
import matplotlib as mpl
lines =['-', '--', '-.', ':']
markers = list(Line2D.markers.keys())

fig,ax = plt.subplots()
for j,level in enumerate(Jmax_list_opt):
    if level>1:
        ax.semilogy(DOFs_opt[j]/DOFs_equi[j],error_list_opt[j],'o', label=r"$ J_\mathrm{max}=%d $"%level)
    
#ax.loglog(eps,eps, 'k--', label = r"$\epsilon$")
ax.set_xlabel(r"compression factor"  )
plt.ylabel(r"$\Vert q-q^{\epsilon,\Delta t,h}\Vert_%d/\Vert q\Vert_%d$"%(norm,norm))
plt.legend(fontsize="small")
plt.grid(which="both",linestyle=':')
save_fig(case+"_errors_vs_DOFS",strict=True)
# plt.xlim([0.5e-4,1])
# plt.ylim([0.5e-5,1])


#tcpu_list_equi[-1] = np.size(dat_list[-1],0)*tcpu_list_equi[-1]/np.size(dat_list_equi[-1],0)
table = np.stack((Jmax_list_opt,DOFs_opt,error_list_opt,tcpu_list_opt,DOFs_equi,error_equi,tcpu_list_equi),axis=1)
table_string = tabulate(table,headers=["$\Jmax$","DOF (adapt)", "rel. error","$\tcpu$ (s)","DOF (equi)", "rel. error","$\tcpu$ (s)"], tablefmt='latex_booktabs',floatfmt=(".0f", ".0f", ".1e", ".0f",".0f", ".1e",".0f"))
print(table_string)
with open(case+"_DOF_error.tabl.tex", "w") as text_file:
    text_file.write(table_string)

# %% spatial convergence for dealiase and without

opt_CDF44_deailas = evaluate_performance("opt_eps_CDF44_dealias*",field_ref, ini="pacman.ini", h5_file_name="q_000001000000.h5", norm=2)
opt_CDF44 = evaluate_performance("opt_eps_CDF44_pacman*",field_ref, ini="pacman.ini", h5_file_name="q_000001000000.h5", norm=2)
equi_CDF44 = evaluate_performance("equi_pacman*",field_ref, ini="pacman.ini", h5_file_name="q_000001000000.h5", norm=2)
equi_CDF44_deailas = evaluate_performance("equi_dealias_pacman*",field_ref, ini="pacman.ini", h5_file_name="q_000001000000.h5", norm=2)

# %%
configs_dicts = {"CDF44 $\epsilon^\\mathrm{opt}$ dealias":opt_CDF44_deailas,
                "equidistant dealias":equi_CDF44_deailas,
                "CDF44 $\epsilon^\\mathrm{opt}$"        :opt_CDF44,
                "equidistant":equi_CDF44}
line_style =["+-","o--","<-",">--"]
dealias = [1,1,0,0]
for k,key in enumerate(configs_dicts):
    print("config:" , key)
    dat = configs_dicts[key]
    dx = np.asarray([1/(2**(j-dealias[k])*(Bs-1)) for j in dat["Jmax"]])
    errors = dat["error"]
    lin = wt.logfit(dx, errors)
    plt.loglog(dx, errors,line_style[k],label=key)
    
plt.loglog(dx,400000*dx**4,'k-',label="$\mathcal{O}({h^4})$")
#wt.add_convergence_labels(dx,400000*dx**4)
plt.legend(loc=2,frameon=False)
plt.xlabel(r"lattice spacing $h$")
plt.ylabel(r"$\Vert q-q^{\epsilon,\Delta t,h}\Vert_%d/\Vert q\Vert_%d$"%(norm,norm))
plt.grid(which="both",linestyle=':')

save_fig("pacman_errors_vs_lattice_spacing",strict=True)

# %% 
