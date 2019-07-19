#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 18:13:54 2019

@author: phil

PLEASE CHANGE ALL Directories here:

"""
from os.path import expanduser
home = expanduser("~")
###############################################################################
# 1.) Location of images saved during post processing
pic_dir = "./images/"           
# 2.) directoy of wabbit-post executable
#wdir = "/home/krah/develop/WABBIT/"++
wdir = home+"/WABBIT/"
# 3.) results to compare data to
resdir_flusi="../../results/cyl_POD/wPOD/vor_crop/"
resdir_flusi_modes="../results/cyl_POD/vorticity_POD2/modes/"
# 4.) Where should wabbit-post save its data
resdir_wPOD_modes=home+"/develop/WABBIT/19.04.19/"
resdir_wabbit =resdir_flusi + "_adapt/"
###############################################################################
eps_list = [100, 10, 1, 0.1, 0.01, 0.001, 0]