#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 11:27:56 2022

@author: pkrah
"""
import numpy as np

def read_performance_file(file_list):
    
    if not isinstance(file_list, list):
        return np.loadtxt(file_list)
        
    perf_dat_list = []
    for file in file_list:
        perf_dat_list.append(np.loadtxt(file))
        
    return perf_dat_list





