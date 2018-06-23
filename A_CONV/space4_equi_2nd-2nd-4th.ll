#!/bin/bash
# submit with: llsubmit timing.ll
# number of nodes:

# @ job_name = WABBIT-convection-dx4
# Type de travail
# @ job_type = mpich
# Nombre de processus MPI demandes
# @ total_tasks = 8
# Permet le passage de total_tasks a mpirun via NB_TASKS
# @ environment = NB_TASKS=$(total_tasks)
# @ output = $(job_name).$(jobid)
# @ error =  $(job_name).$(jobid)
# @ wall_clock_limit = 10:00:00
# @ queue

set -x

export mpi="mpirun -np $NB_TASKS "

if [ ! -f wabbit ]; then
    ln -s ../wabbit
fi

./space4_equi_2nd-2nd-4th.sh
