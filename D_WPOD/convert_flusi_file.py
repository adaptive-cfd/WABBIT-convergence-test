#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 12:16:23 2019

@author: krah

 This script generates all post processing and plots for analyzing 
 the 
    Convert flusi files to WABBIT for different Jmax
       
         

         
"""

import wabbit_tools as wt
import matplotlib.image as img 
import matplotlib.pyplot as plt
import flusi_tools as flusi
import numpy as np
import insect_tools
import os
import farge_colormaps
import glob
from wPODdirs import *
from matplotlib import rc
font = {'family' : 'sans-serif',
        'sans-serif':['Helvetica'],
        'size'   : 22}
rc('font',**font)
rc('text', usetex=True)     


pic_dir = "./images/"
Jmax_list = [  5, 6, 7];
#wdir = "/home/krah/develop/WABBIT/"
wdir = "~/savart/develop/WABBIT/"
resdir_flusi=home+"/develop/results/cyl_POD/wPOD/vor_crop"
resdir_wabbit =resdir_flusi + "_adapt/"
Jmax_dir_list=[resdir_wabbit+"Jmax"+str(Jmax) for Jmax in Jmax_list]






# %% Plot flusi original files:
files = glob.glob(resdir_flusi+'/*.h5')
files.sort()

Nt = len(files)
Npics = 2
iter=0
step=1#Nt//Npics
for Jmax, save_dir in zip(Jmax_list,Jmax_dir_list):

    if not os.path.exists(save_dir):
          os.mkdir(save_dir) # make directory for saving the files

    print("\n\n Converting:",resdir_flusi, " to ", save_dir,"\n\n")
    wt.flusi_to_wabbit_dir(resdir_flusi, save_dir , Jmax, dim=2 )
