#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 18:13:54 2019

@author: phil

PLEASE CHANGE ALL Directories here:

"""
from os.path import expanduser
import numpy as np
home = expanduser("~")
###############################################################################
# 1.) Location of images saved during post processing
pic_dir = "./images/"           
# 2.) directoy of wabbit-post executable
#wdir = "/home/krah/develop/WABBIT/"++
wdir = home+"/WABBIT/"
# 3.) results to compare data to
resdir_flusi="../../results/cyl/wPOD/vor_crop/"
resdir_flusi_modes="../results/cyl/vorticity_POD2/modes/"
# 4.) Where should wabbit-post save its data
resdir_wPOD_modes=home+"/develop/WABBIT/19.04.19/"
resdir_wabbit =resdir_flusi + "_adapt/"
###############################################################################
exponent = np.arange(-5,1)

Jmax_list = [4,5,6]
eps_list = [np.round(val,decimals=-exp) for exp in exponent for val in np.arange(2,11,2)*10.0**(exp)]