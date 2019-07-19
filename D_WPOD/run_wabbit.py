#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 11:16:19 2019

@author: Philipp Krah

This script runs wabbit-post --POD for different parameters of the 
threshold value eps.
The task is to study the dependence of the adaptation on the DECOMPOSITION.

"""

import re
import os
import glob
import numpy as np
from wPODdirs import *
# %% Change parameters here:
eps_list = [100, 10, 1, 0.1, 0.01, 0.001, 0]
eps_list = eps_list[::-1]

wdir = home + "/develop/WABBIT/"                                              # directory of wabbit-post 
data_folder = home + "/develop/results/cyl_POD/wPOD/vor_crop_adapt/"          # datafolder you want to use for POD
data_lists = [data_folder+"/vorx_list.txt"]
mode_lists = ["mode1_list.txt"]
data_file =  glob.glob(data_folder+'/*.h5')[11]
reconstructed_iteration=5
mpicommand = "mpirun -np 4"                                                    # mpi command used for parallel execution
memory ="--memory=16GB"                                                         # how much memory (RAM) available on your pc



# %% run wPOD for different eps:
def run_wPOD_for_different_eps(wdir, data_lists , eps_list, memory, mpicommand, save_log=False):
     
    # if multiple datafiles are given, join the files with spaces:
    data = " ".join(str(data_lists[i]) for i in range(len(data_lists)))
    nc = len(data_lists)
    for eps in eps_list:
        # ------------------------------
        # BUILD EXECUTION COMMAND
        # ------------------------------
        if eps > 0: 
            c = mpicommand + " " +  wdir + \
             "wabbit-post --POD --save_all --nmodes=30 " + memory + \
             " --adapt="+ str(eps) +" --components=" + str(nc) + " --list " + data
             
        else:
            c = mpicommand + " " +  wdir + \
             "wabbit-post --POD --save_all --nmodes=30 " + memory  \
             + " --components=" + str(nc) + " --list " + data
        # pipe output into logfile     
        if save_log:
            c += " > wPOD.log" 
        # -----------------------------
        # Prepare to save data from POD
        # -----------------------------
        save_dir = "eps" + str(eps)
        if not os.path.exists(save_dir):
            os.mkdir(save_dir) # make directory for saving the files
        os.chdir(save_dir)
        
        # -----------------------------
        # Execute Command
        # -----------------------------
        print("\n\n###################################################################")
        print("\t\teps =",eps)
        print("###################################################################")
        print("\n",c,"\n\n")
        success = os.system(c)   # execute command
        os.chdir("../")
        if success != 0:
            print("command did not execute successfully")
            break
    return success

# %% reconstruct for different eps:
def run_wPOD_reconstruction_for_different_eps(wdir, data_lists, eps_list, \
                                              iteration , memory, mpicommand):    
    
    data = " ".join(str(data_lists[i]) for i in range(len(data_lists)))
    command = mpicommand + " " +  wdir + \
             "wabbit-post --POD-reconstruct a_coefs.txt --nmodes=30 " + \
             " --adapt=%s --components=1 --list " + data +" "+ memory \
             + " --timestep="+str(iteration) 
    for eps in eps_list:
        # -----------------------------
        # Prepare to save data from POD
        # -----------------------------
        c = command % str(eps)   # change eps
        c += " > reconstruct.log" 
        save_dir = "eps" + str(eps)
        if not os.path.exists(save_dir):
            os.mkdir(save_dir) # make directory for saving the files
        os.chdir(save_dir)
        # generate lists of POD modes needed for reconstruction
        success=generate_wPODmode_list_for_eps_dir( '.' )
        if not success:
            print('no mode lists generated for eps=', eps)
            break
        # -----------------------------
        # Execute Command
        # -----------------------------
        print("\n\n###################################################################")
        print("\t\teps =",eps)
        print("###################################################################")
        print("\n",c,"\n\n")
        success = os.system(c)   # execute command
        os.chdir("../")
        if success != 0:
            print("command did not execute successfully")
            break
        return success
    
    
# % generate mode list for different eps
def generate_wPODmode_list_for_eps_dir( eps_dir ):
    import re
    
    h5file_names = glob.glob(eps_dir+'/[PODmode,mode]*.h5')
    h5file_names.sort()
    
    n_component = 0
    while len(h5file_names)>0:
        # make a list of all h5files which contain "mode[digit]" in their name
        # Here digit is the number of the component
        n_component += 1
        regexp = re.compile('[PODmode,mode]['+str(n_component)+']?_')
        mode_list = [h5file for h5file in h5file_names if regexp.search(h5file)]
        
        fpointer = open("mode"+str(n_component)+"_list.txt","w") 
        for element in mode_list: 
            # remove the elements of the specific component from the list    
            h5file_names.remove(element)
            # write all modes of the component to to file
            absfilepath = os.path.abspath(element) + '\n'
            print(absfilepath)
            fpointer.write(absfilepath)
        fpointer.close()
        
    ## if successfull then:
    return n_component>0

# %% adapt one snapshot for different eps: logfile written in adapt .log
def adpatation_for_different_eps(wdir, h5_fname, eps_list, memory, mpicommand, create_log_file=True):
    
    # generate new file names
    file = h5_fname.split("/")[-1]
    filesparse =file.replace('_','-sparse_')
    filedense =file.replace('_','-dense_')
    
    # command for sparsing file
    command = mpicommand + " " +  wdir + \
             "wabbit-post --dense-to-sparse --eps=%s " + memory + \
              " "+ file
    # command for densing file again
    dense_command = mpicommand + " " +  wdir + \
             "wabbit-post --sparse-to-dense " + \
              " "+ filesparse + " "+filedense + ">>adapt.log"
              
    for eps in eps_list:
        # -----------------------------
        # Prepare to save data from POD
        # -----------------------------
        c = command % str(eps)   # change eps
        if create_log_file:
            c += " > adapt.log" 
        save_dir = "eps" + str(eps)
        if not os.path.exists(save_dir):
            os.mkdir(save_dir) # make directory for saving the files
        os.chdir(save_dir)
        
        # copy given file in current directory 
        os.system("cp "+h5_fname+ " .")
        # -----------------------------
        # Execute Command
        # -----------------------------
        print("\n\n###################################################################")
        print("\t\teps =",eps)
        print("###################################################################")
        print("\n",c,"\n\n")
        success = os.system(c)   # execute command
        os.rename(file,filesparse) # rename file
        print("\n",dense_command,"\n\n")
        success += os.system(dense_command) # dense the file for comparison to original
        os.chdir("../")
        if success != 0:
            print("command did not execute successfully")
            break
    return success



# %% run scripts:
#run_wPOD_for_different_eps(wdir, data_lists, eps_list, memory, mpicommand, save_log=True)
#run_wPOD_reconstruction_for_different_eps(wdir, mode_lists, eps_list, \
#                                          reconstructed_iteration, memory, mpicommand)
#data_file = np.sort(glob.glob(data))[100]
adpatation_for_different_eps(wdir, data_file, eps_list[1:], memory, mpicommand)
