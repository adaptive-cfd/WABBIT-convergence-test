#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 18:13:54 2019

@author: phil

PLEASE CHANGE ALL Directories here:

"""
from os.path import expanduser
import numpy as np
from matplotlib import rc

###############################################################################
# Directories
###############################################################################
home = expanduser("~")
# 1.) Location of images saved during post processing
pic_dir = "./"           
# 2.) directoy of wabbit-post executable
#wdir = "/home/krah/develop/WABBIT/"++
wdir = home+"/develop/WABBIT/"
# 3.) directoy of wabbit-post executable
work = home+"/develop/WPOD/D_WPOD/"

dirs= {
       'wabbit' : wdir ,    # directory of wabbit
       'work'   : work,     # where to save big data files
       'images' : pic_dir  # pictures are saved here
       }


wabbit_setup = {
        'mpicommand' : "mpirun --use-hwthread-cpus -np 2",
        'memory'     : "--memory=16GB"
        }
###############################################################################
# LATEX FONT
###############################################################################
font = {'family' : 'serif',
        'size'   : 16}
rc('font',**font)
rc('text', usetex=True)   
