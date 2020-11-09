#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 10:55:09 2019

@author: krah
"""
import numpy as np
import os
from wPODdirs import *
import glob
from wPOD import synthetic_test_case, init
from wPOD import adapt2eps
from wPOD import plot_modes
from wPOD.wPODerror import *
from wPOD.run_wabbit import *
import wabbit_tools as wt
import time
font = {'family' : 'serif',
        'size'   : 20}
rc('font',**font)
rc('text', usetex=True)   


# %% Set Parameters
############################################################################### 
class params:
   case = "Bumbs" # choose [Bumbs,Mendez] 
   wavelets_norm = "L2"
   wavelets = "CDF44"
   slow_singular_value_decay=False
   #eps_list  = np.asarray([0]+[float("%1.1e" %eps) for eps in np.logspace(-12,0,6)])  # threshold of adaptation
   eps_list  = np.asarray([0]+[float("%1.1e" %eps) for eps in np.logspace(-5,0,10)])
   #eps_list = np.delete(eps_list,1)
   jmax_list = [4,5,6]        # maximal tree level
   Nt= 2**7
   target_rank = 30
   
   class domain:
        N = [2**10, 2**10]      # number of points
        L = [30, 30]            # length of domain
  
###############################################################################
# directories needed
dir_list=dirs.copy()     
if params.slow_singular_value_decay:
    dir_list["work"]=dirs["work"]+"/bump/CDF44c/01/"
    dir_list["images"]=dirs["images"]+"/bump/CDF44c/01/"
else:
    dir_list["work"]=dirs["work"]+"/bump/CDF44c/02/"
    dir_list["images"]=dirs["images"]+"/bump/CDF44c/02/"

os.makedirs(dir_list["images"], exist_ok=True)
os.makedirs(dir_list["work"], exist_ok=True)

Jmax_dir_list = [ "Jmax"+str(Jmax)+"/" for Jmax in params.jmax_list]
eps_list = params.eps_list
eps_dir_list = [ "eps%1.1e" %eps for eps in eps_list]

# %%

file_list = synthetic_test_case(dir_list, params, wabbit_setup)
    
data = {"folder" :   dir_list["work"],
        "qname" : ["u"]
        }
mode_lists = ["mode1_list.txt"]
run_wabbit_POD(wabbit_setup, dir_list, data, params.jmax_list, params.eps_list, mode_lists,\
                reconstructed_iteration=10,n_modes=params.target_rank,  wavelets=params.wavelets)
# %%

delta_err,clist=plot_wPODerror(params.jmax_list, Jmax_dir_list,eps_dir_list, \
                                eps_list, dir_list, n_star=30,eps_list_plot=eps_list,\
                                show_legend=False, show_title=False, alternate_markers=True)

# %% Adaption test 
wdir = dir_list["wabbit"]
plt.close("all")
fig_adapt = [plt.figure(38),plt.figure(39), plt.figure(37)]
ax_adapt = [fig.add_subplot() for fig in fig_adapt]
eps_list_  = np.asarray([0]+[float("%1.1e" %eps) for eps in np.logspace(-15,1,20)])  # thre
for k,jmax in enumerate(params.jmax_list):
    markers = ['o', 'x', '<', 'v', '^', '+', '>', 's', 'd']
    h5_fname = sorted(glob.glob(dir_list["work"]+"Jmax%d"%jmax+'/*.h5'))[-1]
    [l2error, linferror, Nblocks, Nblocksdense]=adapt2eps(h5_fname, \
    wdir, eps_list_, wabbit_setup['memory'], 'mpirun -np 6', \
    normalization = params.wavelets_norm,create_log_file=True, show_plot=False,
    wavelets=params.wavelets)
    Ne = 1
    print("Number of blocks Jmax=%d : "%jmax, Nblocks)
    nm = np.mod(k,np.size(markers))
    #### plot compression error
    ax_adapt[0].loglog(eps_list_[Ne:],l2error[Ne:],linestyle='-',marker = markers[nm],label="$J_{\\mathrm{max}}=%d$"%jmax)
    #linfplt, = ax_adapt[0].loglog(eps_list[Ne:],linferror[Ne:],'-.*', label= "$\Vert u(x) - [u(x)]^\epsilon \Vert_\infty$")
    ####  plot compression rate
    ax_adapt[1].semilogx(eps_list_[Ne:],Nblocks[Ne:]/Nblocksdense, \
            linestyle='-',marker = markers[nm],label="$J_{\\mathrm{max}}=%d$"%jmax)
    ax_adapt[2].semilogy(Nblocks[Ne:]/Nblocksdense, l2error[Ne:], \
            linestyle='-',marker = markers[nm],label="$J_{\\mathrm{max}}=%d$"%jmax)


# Create a legend for the first line.
ax_adapt[0].loglog(eps_list_,eps_list_, 'k--',label="$\epsilon$")
ax_adapt[0].grid(which='both',linestyle=':')
ax_adapt[0].set_xlabel("$\epsilon$", fontsize=25)
ax_adapt[0].set_ylabel("$\\mathcal{E}_{\\mathrm{wavelet}}$", fontsize=25)

ax_adapt[1].grid(which='both',linestyle=':')
ax_adapt[1].set_xlabel("$\epsilon$", fontsize=25)
ax_adapt[1].set_ylabel("Compression Factor $C_{\mathrm{f}}$", fontsize=25)
ax_adapt[0].set_xlim=ax_adapt[1].get_xlim()

ax_adapt[2].legend()
ax_adapt[2].grid(which='both',linestyle=':')
ax_adapt[2].set_ylabel("$\\mathcal{E}_{\\mathrm{wavelet}}$", fontsize=25)
ax_adapt[2].set_xlabel("Compression Factor $C_{\mathrm{f}}$", fontsize=25)

ax_adapt[0].legend()
ax_adapt[1].legend()
ax_adapt[2].legend()

fig_adapt[0].savefig(dir_list["images"]+ 'compression_err4th.svg', dpi=300, transparent=True, bbox_inches='tight' )# -*- coding: utf-8 -*-
fig_adapt[1].savefig(dir_list["images"]+ 'compression_rate.svg', dpi=300, transparent=True, bbox_inches='tight' )  
fig_adapt[2].savefig(dir_list["images"]+ 'compression_err_vs_rate.svg', dpi=300, transparent=True, bbox_inches='tight' )  

fig_adapt[0].savefig(dir_list["images"]+ 'compression_err4th.pdf', dpi=300, transparent=True, bbox_inches='tight' )# -*- coding: utf-8 -*-
fig_adapt[1].savefig(dir_list["images"]+ 'compression_rate.pdf', dpi=300, transparent=True, bbox_inches='tight' )  
fig_adapt[2].savefig(dir_list["images"]+ 'compression_err_vs_rate.pdf', dpi=300, transparent=True, bbox_inches='tight' )  

    
# %% Plot some modes

plt.close("all")
quantity="mode1"
i = 4 
eps_dir = eps_dir_list[i]
Jmax_dir = Jmax_dir_list[-1]
files = glob.glob(dir_list["work"]+Jmax_dir+eps_dir+'/'+quantity+'_*.h5')
files.sort()
acoef_file = dir_list["work"]+Jmax_dir+eps_dir+"/a_coefs.txt"
plot_modes(files[:10], acoef_file, quantity,quantity+"-epsilon%1.1e"%eps_list[i],pic_dir=dir_list["images"])

# %% Plot some reconstructions
os.makedirs(dir_list["images"], exist_ok=True)
plt.close("all")
quantity=""
i = 1
eps_dir = eps_dir_list[i]
Jmax_dir = Jmax_dir_list[-2]
files = glob.glob(dir_list["work"]+Jmax_dir+eps_dir+'/'+quantity+'_*.h5')
plot_modes(files[:5],quantity,quantity+"-epsilon%1.1e"%eps_list[i],pic_dir=dir_list["images"])


# %% compare randomiced SVD to wPOD 
from numpy.linalg import svd
from wPOD.randomSVD import rSVD
# 
dim = len(params.domain.L)
x = [np.linspace(0,params.domain.L[d],params.domain.N[d]) \
     for d in range(dim) ]
t = np.linspace(0,2*np.pi,params.Nt)
[X,Y] = np.meshgrid(*x)

phi_matrix, amplitudes,lambd = init([X,Y],t,params.domain.L,case=params.case, slow_decay=params.slow_singular_value_decay)
   
phi = np.reshape(phi_matrix,[*params.domain.N,params.Nt])

Nsnapshots = phi_matrix.shape[1]

ts = time.time()
[U, S, VT] = svd(phi_matrix,0)
t_svd = time.time() - ts
print("svd time: %2.2f sec"% t_svd)

error_svd=[np.linalg.norm(phi_matrix-U[:,:r]@np.diag(S[:r])@VT[:r,:],ord=2)/S[0] for r in range(params.target_rank+1)]

# %%

rank = params.target_rank
nr_power_iter = 0
nr_oversampl  = 5

ts = time.time()
[rU,rS,rVT,Psize] = rSVD(phi_matrix, r = rank , 
                   q = nr_power_iter, p = nr_oversampl)
t_rsvd = time.time() - ts
print("rsvd time: %2.2f sec, Problem size: %2d "% (t_rsvd,Psize))

error_rsvd=[np.linalg.norm(phi_matrix-rU[:,:r]@np.diag(rS[:r])@rVT[:r,:],ord=2)/S[0] for r in range(np.size(rVT,0)+1)]



# %% wPOD error vs number of blocks
from matplotlib.ticker import NullFormatter

plt.close("all")
wPODerr_dict,PODerr_dict, delta_PODerr = read_wPOD_error(params,dir_list)

wPODerr = wPODerr_dict[params.jmax_list[0],params.eps_list[1]]
PODerr = PODerr_dict[params.jmax_list[0],params.eps_list[1]]


if params.slow_singular_value_decay:
    labl_lambd="$\exp(-k/100)$"
    zoom_loc = [0.1, 0.1, 0.4, 0.4]
else:
    labl_lambd="$\exp(-k/3)$"
    zoom_loc = [0.05, 0.1, 0.4, 0.4]

# PLOT eigenvalues
fig=plt.figure(34)
ax = fig.add_subplot()
ax.semilogy((S/S[0])**2,'o',label="SVD", markersize=2)
ax.semilogy(PODerr,'x',label="wPOD",markersize=2)    
ax.semilogy(lambd**2,'k-',label=labl_lambd, markersize=2)
#ax.semilogy(np.sqrt(wPODerr), linestyle='--', label="wPOD tot")
#ax.semilogy(error_rsvd,':',label="random tot")
ax.semilogy((rS/rS[0])**2,'+', label="rSVD",markersize=2)
ax.set_ylabel("eigenvalues $\lambda_k/\lambda_1$")
ax.set_xlabel("$k$")
ax.set_ylim([1e-17,10])
ax.set_xlim([0,50])
lg=plt.legend(loc=4,fontsize=18)
lg.legendHandles[0]._legmarker.set_markersize(6)
lg.legendHandles[1]._legmarker.set_markersize(6)
lg.legendHandles[2]._legmarker.set_markersize(6)
lg.legendHandles[3]._legmarker.set_markersize(6)

# inset axes....
#axins = ax.inset_axes(zoom_loc)
#axins.semilogy((S/S[0])**2,'o',fillstyle="none")
#axins.semilogy(PODerr,'x',label="wPOD")
#axins.semilogy(lambd**2,'k-',label=labl_lambd)
#axins.semilogy((rS/rS[0])**2,'+', label="rSVD")
#axins.yaxis.set_major_formatter(NullFormatter())
#axins.yaxis.set_minor_formatter(NullFormatter())
## sub region of the original image
#
#x1, x2, y1, y2 = 0, params.target_rank+15, (S[params.target_rank+15]/S[0])**2, (S[0]/S[0])**2
#axins.set_xlim(x1, x2)
#axins.set_ylim(y1, y2)
#axins.set_xticks([0,10,20,30,40])
#axins.set_yticks([])
##axins.set_xticklabels('')
#axins.set_yticklabels('')
#ax.indicate_inset_zoom(axins)

plt.show()
fig.savefig(dir_list["images"]+ 'lambda_wPODvsrSVD.png', dpi=300, transparent=True, bbox_inches='tight' )
fig.savefig(dir_list["images"]+ 'lambda_wPODvsrSVD.svg', dpi=300, transparent=True, bbox_inches='tight' )

# PLOT total error
fig=plt.figure(120)
ax = fig.add_subplot()
ax.semilogy((S/S[0])**2,'o',label="SVD",markersize=2)
ax.semilogy(wPODerr,'x',label="wPOD",markersize=2)
ax.semilogy(lambd**2,'k-',label=labl_lambd, markersize=2)
ax.semilogy((np.asarray(error_rsvd))**2,'+', label="rSVD",markersize=2)
ax.set_ylabel("$\mathcal{E}_{\\mathrm{(r)SVD/wPOD}}$")
ax.set_xlabel("$k$ modes")
ax.set_ylim([1e-17,10])
ax.set_xlim([0,50])
lg=plt.legend(loc=4,fontsize=18)
lg.legendHandles[0]._legmarker.set_markersize(6)
lg.legendHandles[1]._legmarker.set_markersize(6)
lg.legendHandles[2]._legmarker.set_markersize(6)
lg.legendHandles[3]._legmarker.set_markersize(6)

# inset axes....
#axins2 = ax.inset_axes(zoom_loc)
#pl=axins2.semilogy((S/S[0])**2,'o',fillstyle="none")
#axins2.semilogy(wPODerr,'x',label="wPOD",)
#axins2.semilogy(lambd**2,'k-',label=labl_lambd, markersize=2)
#axins2.semilogy((np.asarray(error_rsvd))**2,'+', label="rSVD") 
## sub region of the original image
#x1, x2, y1, y2 = 0, params.target_rank+15, (S[params.target_rank+15]/S[0])**2, (S[0]/S[0])**2
#axins2.set_xlim(x1, x2)
#axins2.set_ylim(y1, y2)
#axins2.set_xticks([0,10,20,30,40])
#axins2.yaxis.set_major_formatter(NullFormatter())
#axins2.yaxis.set_minor_formatter(NullFormatter())
#ax.indicate_inset_zoom(axins2)

plt.show()
fig.savefig(dir_list["images"]+ 'error_wPODvsrSVD.svg', dpi=300, transparent=True, bbox_inches='tight' )
fig.savefig(dir_list["images"]+ 'error_wPODvsrSVD.png', dpi=300, transparent=True, bbox_inches='tight' )
 # %%
def fetch_wPOD_params(params,dirs):
    import re
    import numpy as np
    
    def read_log(logfile):
        with open(logfile) as flog:
            for line in flog:
               
                if " Bs        =" in line:
                    Bs=[ int(b) for b in re.findall(r'\d+', line)]
                    Bs = np.asarray(Bs)
                if "Nblocks (if all trees dense)=" in line:
                    Nblocks_dense=int(re.findall(r'\d+', line)[0])
                if "Nblocks used (sparse)=" in line:
                    Nblocks_sparse=int(re.findall(r'\d+', line)[0])
        
        return Bs, Nblocks_sparse, Nblocks_dense
    eps_list = params.eps_list
    eps_list = eps_list[np.where(eps_list>0)]
    params.Bs_list=np.zeros([len(params.jmax_list), 3]) # number of block points
    params.Nbsparse_list=np.zeros([len(params.jmax_list),len(eps_list)]) # number of blocks
    params.Nbdense_list=np.zeros([len(params.jmax_list),len(eps_list)]) # number of blocks
    params.Np_list=np.zeros([len(params.jmax_list),len(eps_list)]) # number of total points
    for n, Jmax in enumerate(params.jmax_list):
        for m, eps in enumerate(eps_list):
            logfile = dirs["work"]+"Jmax%d"%Jmax+"/eps%1.1e"%eps + "/wPOD.log"
            Bs, Nblocks_sparse, Nblocks_dense = read_log(logfile)
            params.Bs_list[n,:]= Bs
            params.Nbsparse_list[n,m]= Nblocks_sparse
            params.Nbdense_list[n,m]= Nblocks_dense
            params.Np_list[n,m]= np.prod(Bs[:2]-1)*Nblocks_sparse
            

# %%
fetch_wPOD_params(params,dir_list)
delta_err_rsvd = [np.abs(error_rsvd[k]**2-error_svd[k]**2) for k in range(params.target_rank+1)]
fig=plt.figure(35)
ax = fig.add_subplot()

markers=['o', '+', 'x', '*', '.', 'X']
for n, Jmax in enumerate(params.jmax_list[:2]):
    delta_err_wPOD = [delta_PODerr[Jmax,eps][params.target_rank] for eps in params.eps_list[np.where(eps_list>0)]]            
    ax.loglog(params.Np_list[n,:]/np.size(rU), delta_err_wPOD, \
              label="$J_{\\mathrm{max}} =%d $"%Jmax, marker=markers[n])

xlim=ax.get_xlim()
ax.hlines(delta_err_rsvd[params.target_rank],0,ax.get_xlim()[1]*10,linestyle="--",label="rSVD ($q=%d$)"%(nr_oversampl+params.target_rank))
ax.set_xlim(xlim)
ax.legend()
ax.set_xlabel("memory efficiency $\\frac{S_\\mathrm{wPOD}}{S_\\mathrm{rSVD}}$")
ax.set_ylabel("$\mid \mathcal{E}_{\mathrm{POD}}(r^*,0)-\mathcal{E}_{\mathrm{wPOD}}(r^*,\epsilon)\mid$")
fig.savefig(dir_list["images"]+ 'memory_wPODvsrSVD.png', dpi=300, transparent=True, bbox_inches='tight' )  
fig.savefig(dir_list["images"]+ 'memory_wPODvsrSVD.svg', dpi=300, transparent=True, bbox_inches='tight' )  
