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
###############################################################################

# %% run wPOD for different eps:
def run_wPOD_for_different_eps(dirs, data_lists , eps_list, Jmax, memory, \
                               mpicommand, save_log=False, n_modes=30, wavelets="CDF40",\
                                normalization="L2"):
    # n_modes ... number of modes created in algorithm
    # if multiple datafiles are given, join the files with spaces:

    work = dirs["work"]
    wdir = dirs["wabbit"]
    data = " ".join(str(data_lists[i]) for i in range(len(data_lists)))
    nc = len(data_lists)
    Jmaxdir = work+"/Jmax"+str(Jmax)
    if not os.path.exists(Jmaxdir):
            os.makedirs(Jmaxdir, exist_ok=True) # make directory for saving the files
    os.chdir(Jmaxdir)

    for eps in eps_list:
        # ------------------------------
        # BUILD EXECUTION COMMAND
        # ------------------------------
        if eps > 0:
            c = mpicommand + " " +  wdir + \
             "wabbit-post --POD --save_all --nmodes="+str(n_modes)+" " + memory \
             + " --adapt=%1.1e"%eps +" --components=" + str(nc) + ' --list="' + data + '"' \
             + " --order="+wavelets + " --eps-norm="+ normalization

        else:
            c = mpicommand + " " +  wdir + \
             "wabbit-post --POD --save_all --nmodes="+str(n_modes)+" " + memory  \
             + " --components=" + str(nc) + ' --list="' + data+'"'\
             + " --order="+wavelets + " --eps-norm="+ normalization
        # pipe output into logfile
        if save_log:
            c += " > wPOD.log"
        # -----------------------------
        # Prepare to save data from POD
        # -----------------------------
        save_dir = "eps%1.1e"%eps
        if not os.path.exists(save_dir):
            os.mkdir(save_dir) # make directory for saving the files
        os.chdir(save_dir)

        # -----------------------------
        # Execute Command
        # -----------------------------
        print("\n\n###################################################################")
        print("\t\teps =%1.1e"%eps)
        print("###################################################################")
        print("\n",c,"\n\n")
        success = os.system(c)   # execute command
        os.chdir("../")
        if success != 0:
            print("command did not execute successfully")
            break
    # go back to original directory
    os.chdir("../")
    return success

# %% reconstruct for different eps:
def run_wPOD_reconstruction_for_different_eps(wdir, mode_lists, eps_list, Jmax, \
                                              iteration, memory, mpicommand, \
                                            workdir = "./",  wavelets="CDF40",\
                                            normalization="L2"):

    data = " ".join(str(mode_lists[i]) for i in range(len(mode_lists)))
    command = mpicommand + " " +  wdir + \
             "wabbit-post --POD-reconstruct --time_coefficients=a_coefs.txt --nmodes=30 " + \
             ' --adapt=%1.1e --components=1 --mode-list="' + data +'" '+ memory \
             + " --iteration="+str(iteration) + " --order="+wavelets + " --eps-norm="+ normalization

    Jmaxdir = workdir+"/Jmax"+str(Jmax)
    if not os.path.exists(Jmaxdir):
            os.mkdir(Jmaxdir) # make directory for saving the files
    os.chdir(Jmaxdir)
    for eps in eps_list:
        # -----------------------------
        # Prepare to save data from POD
        # -----------------------------
        c = command % eps   # change eps
        c += " > reconstruct.log"
        save_dir = "eps%1.1e"%eps
        if not os.path.exists(save_dir):
            os.mkdir(save_dir) # make directory for saving the files
        os.chdir(save_dir)
        # generate lists of POD modes needed for reconstruction
        success=generate_list_of_name_in_dir( 'mode',  '.' )
        if not success:
            print('no mode lists generated for eps=', eps)
            break
        # -----------------------------
        # Execute Command
        # -----------------------------
        print("\n\n###################################################################")
        print("\t\teps =%1.1e"%eps)
        print("###################################################################")
        print("\n",c,"\n\n")
        success = os.system(c)   # execute command
        os.chdir("../")
        if success != 0:
            print("command did not execute successfully")
            break

    # go back to original directory
    os.chdir("../")
    return success

# %% reconstruct for different eps:
def run_wPODerr_for_different_eps(wdir, data_lists, mode_lists, eps_list,  Jmax , \
                iteration, memory, mpicommand, workdir = "./", wavelets="CDF40",\
                                normalization="L2"):

    # in the first step we make a concatenatet string form all elements in the
    # snapshot and mode lists
    data = " ".join(str(data_lists[i]) for i in range(len(data_lists)))
    modes = " ".join(str(mode_lists[i]) for i in range(len(mode_lists)))

    n_components = len(data_lists)
    # command executed for every eps
    command = mpicommand + " " \
             +  wdir + "wabbit-post --POD-error --time_coefficients=a_coefs.txt "  \
             + " --adapt=%1.1e --components=" + str(n_components) \
             + ' --snapshot-list="' + data \
             + '" --mode-list="'     + modes\
             + '" '+ memory \
             + " --iteration=" + str(iteration) \
             + " --order="+wavelets + " --eps-norm="+ normalization

    # generate new Jmaxdir if not exist
    Jmaxdir = workdir+"/Jmax"+str(Jmax)
    if not os.path.exists(Jmaxdir):
            os.mkdir(Jmaxdir) # make directory for saving the files
    os.chdir(Jmaxdir)

    # loops over all eps-thresholds
    for eps in eps_list:
        # -----------------------------
        # Prepare to save data from POD
        # -----------------------------
        c = command % eps   # change eps
        c += " > wPODerror.log"

        save_dir = "eps%1.1e"%eps
        if not os.path.exists(save_dir):
            os.mkdir(save_dir) # make directory for saving the files
        os.chdir(save_dir)

        # generate lists of POD modes needed for reconstruction
        success=generate_list_of_name_in_dir( 'mode',  '.', abspath=False)
        if not success:
            print('no mode lists generated for eps=', eps)
            break

        # -----------------------------
        # Execute Command
        # -----------------------------
        print("\n\n###############################################################")
        print("\t\teps =%1.1e"%eps)
        print("###################################################################")
        print("\n",c,"\n\n")
        success = os.system(c)   # execute command
        os.chdir("../")
        if success != 0:
            print("command did not execute successfully")
            break

    # go back to original directory
    os.chdir("../")
    return success

# %% generate mode list for different eps
def generate_list_of_name_in_dir( name, directory, save_dir=None, abspath=True ):
    import re

    h5file_names = glob.glob(directory+'/'+name+'*.h5')
    h5file_names.sort()

    n_component = 0
    fname_list= []

    print(" I will generate a list of files containing: " + name
          + " in the directory: ", directory )

    while len(h5file_names)>0:
        # make a list of all h5files which contain "mode[digit]" in their name
        # Here digit is the number of the component
        n_component += 1
        regexp = re.compile(name+'['+str(n_component)+']?_')
        mode_list = [h5file for h5file in h5file_names if regexp.search(h5file)]
        if save_dir:
            fname_list = save_dir+"/"+name+str(n_component)+"_list.txt"
            fpointer = open(fname_list , 'w')
        else:
            fname_list = name+str(n_component) + "_list.txt"
            fpointer = open(fname_list , "w")

        for element in mode_list:
            # remove the elements of the specific component from the list
            h5file_names.remove(element)
            # write all modes of the component to to file
            if abspath: 
                filepath = os.path.abspath(element) + '\n'
            else:
                filepath = os.path.relpath(element,'.') + '\n'
            #print(absfilepath)
            fpointer.write(filepath)
        fpointer.close()

    ## if successfull then:
    return n_component>0,fname_list

# %% adapt one snapshot for different eps: logfile written in adapt .log
def adaptation_for_different_eps(wdir, data_folder, eps_list, Jmax, memory, \
                                 mpicommand, create_log_file=True,  wavelets="CDF40",\
                                normalization="L2"):

    # generate new file names
    h5_fname =  glob.glob(data_folder+'/Jmax'+str(Jmax)+'/*.h5')[11]

    print( "Input data file: ", h5_fname, "\n")
    file = h5_fname.split("/")[-1]
    filesparse =file.replace('_','-sparse_')
    filedense =file.replace('_','-dense_')
    # command for sparsing file
    command = mpicommand + " " +  wdir + \
             "wabbit-post --dense-to-sparse --eps=%1.1e" + " --order="+wavelets \
                 + " --eps-norm="+ normalization+ memory + \
              " "+ filesparse
    # command for densing file again
    dense_command = mpicommand + " " +  wdir + \
             "wabbit-post --sparse-to-dense " + \
              " "+ filesparse + " "+filedense + ">>adapt.log"
    Jmaxdir = "Jmax"+str(Jmax)
    if not os.path.exists(Jmaxdir):
            os.mkdir(Jmaxdir) # make directory for saving the files
    for eps in eps_list:
        # -----------------------------
        # Prepare to save data from POD
        # -----------------------------
        c = command % eps   # change eps
        if create_log_file:
            c += " > adapt.log"
        save_dir =  Jmaxdir+"/eps%1.1e" + eps
        if not os.path.exists(save_dir):
            os.mkdir(save_dir) # make directory for saving the files
        os.chdir(save_dir)

        # copy given file in current directory
        os.system("cp "+h5_fname+ " .")
        os.system("cp "+h5_fname+ " ./" + filesparse)
        # -----------------------------
        # Execute Command
        # -----------------------------
        print("\n\n###################################################################")
        print("\t\teps =%1.1e"%eps)
        print("###################################################################")
        print("\n",c,"\n\n")
        success = os.system(c)   # execute command

        print("\n",dense_command,"\n\n")
        success += os.system(dense_command) # dense the file for comparison to original
        os.chdir("../../")
        if success != 0:
            print("command did not execute successfully")
            break
    return success

# %% adapt one snapshot for different eps: logfile written in adapt .log
    

def compute_vorabs(wdir, uxfile, uyfile, memory, mpicommand, uzfile=""):

    # command for sparsing file
    if not uzfile:
        command = mpicommand + " " +  wdir + \
             "wabbit-post --vor-abs "+ uxfile + " " + uyfile + " 4" 
    else:
        command = mpicommand + " " +  wdir + \
             "wabbit-post --vor-abs "+ uxfile + " " + uyfile + " " + uzfile + " 4" 


    print("\n", command,"\n\n")
    success = os.system(command)   # execute command
    
    if success != 0:
        print("command did not execute successfully")

    return success

def compute_vorticity(wdir, uxfile, uyfile, memory, mpicommand, uzfile=""):

    # command for sparsing file
    if not uzfile:
        command = mpicommand + " " +  wdir + \
             "wabbit-post --vorticity "+ uxfile + " " + uyfile + " 4" 
    else:
        command = mpicommand + " " +  wdir + \
             "wabbit-post --vorticity "+ uxfile + " " + uyfile + " " + uzfile + " 4" 


    print("\n", command,"\n\n")
    success = os.system(command)   # execute command
    
    if success != 0:
        print("command did not execute successfully")

    return success

# %% run scripts:

def run_wabbit_POD(wabbit_setup, dirs, data, Jmax_list, eps_list, mode_lists, \
                   reconstructed_iteration, n_modes=30, wavelets="CDF44",normalization="L2"):

    wdir = dirs["wabbit"]
    work = dirs["work"]
    memory = wabbit_setup["memory"]
    mpicommand = wabbit_setup["mpicommand"]
    data_folder = data["folder"]
    qnames = data["qname"]
    for Jmax in Jmax_list:
        folder = data_folder + "/Jmax" + str(Jmax) +"/"
        data_lists=[]
        for qname in qnames:
            success,data_list= generate_list_of_name_in_dir( qname,folder,save_dir = folder )
            data_lists.append(data_list)
            if not success:
                print("failed to construct file list")
                return

        success = 0
        success += run_wPOD_for_different_eps(dirs, data_lists, eps_list,\
                                              Jmax, memory, mpicommand, save_log=True, \
                                               n_modes=n_modes, wavelets=wavelets, \
                                               normalization=normalization)
        if success!=0:
            print("wPOD did not execute successfully")
            break

        #success += run_wPOD_reconstruction_for_different_eps(wdir, mode_lists, eps_list, Jmax, \
        #                                     reconstructed_iteration, memory, mpicommand, workdir = work)
        if success!=0:
            print("wPOD reconstruction did not execute successfully")
            break

        success += run_wPODerr_for_different_eps(wdir, data_lists, mode_lists, eps_list,  Jmax , \
                   reconstructed_iteration, memory, mpicommand,workdir = work, wavelets=wavelets, \
                                              normalization=normalization)
        if success!=0:
            print("error estimation did not execute successfully")
            break

        #success += adaptation_for_different_eps(wdir, data_folder, eps_list, Jmax, memory, mpicommand)
        if success!=0:
            print("adaptation did not execute successfully")
            break
