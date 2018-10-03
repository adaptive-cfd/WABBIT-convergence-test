#!/usr/bin/env python3

import numpy as np
import wabbit_tools
import insect_tools
import glob
import configparser
import datetime
import os
import sys

class bcolors:
        HEADER = '\033[95m'
        OKBLUE = '\033[94m'
        OKGREEN = '\033[92m'
        WARNING = '\033[93m'
        FAIL = '\033[91m'
        ENDC = '\033[0m'
        BOLD = '\033[1m'
        UNDERLINE = '\033[4m'


print("----------------------------------------")
print(" Remaining walltime estimator for wabbit")
print("----------------------------------------")

if len(sys.argv) > 1:
    dir = sys.argv[1]
else:
    dir = './'


if dir[-1] == '/':
    dir = dir
else:
    dir = dir + '/'


if not os.path.isfile(dir + 'timesteps_info.t'):
    raise ValueError("The file timesteps_info.t has not been found here.")

# load the data file
d = insect_tools.load_t_file(dir + 'timesteps_info.t')

# look for the ini file, this gives us the information at what time the run is done
inifile = glob.glob(dir + '*.ini')

if (len(inifile) > 1):
    raise ValueError('ERROR MORE THAN ONE INI FILE in this directory.')


print("We found and extract the final time in the simulation from: "+inifile[0])
T = wabbit_tools.get_ini_parameter( inifile[0], 'Time', 'time_max', float)

# how many time steps did we already do?
nt_now = d.shape[0]

# avg CPU second for this run
tcpu_avg = np.mean( d[:,1] )

# avg time step until now
dt = d[-1,0] / nt_now

# how many time steps are left
nt_left = (T-d[-1,0]) / dt

# this is what we have to wait still
time_left = round(nt_left * tcpu_avg)

print("Time to reach: T=%e. Now: we did nt=%i to reach T=%e and the remaing time is: %s%s%s"
      % (T, nt_now, d[-1,0], bcolors.OKGREEN, str(datetime.timedelta(seconds=time_left)), bcolors.ENDC ) )

