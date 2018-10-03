#!/usr/bin/env python3

import numpy as np
import wabbit_tools
import insect_tools
import glob
import configparser
import datetime
import os

class bcolors:
        HEADER = '\033[95m'
        OKBLUE = '\033[94m'
        OKGREEN = '\033[92m'
        WARNING = '\033[93m'
        FAIL = '\033[91m'
        ENDC = '\033[0m'
        BOLD = '\033[1m'
        UNDERLINE = '\033[4m'


# look for the ini file, this gives us the information at what time the run is done
inifile = glob.glob('*.ini')

if (len(inifile) > 1):
    raise ValueError('ERROR MORE THAN ONE INI FILE in this directory.')


print("We found and check the INI file: "+inifile[0])
wabbit_tools.check_parameters_for_stupid_errors( inifile[0] )
