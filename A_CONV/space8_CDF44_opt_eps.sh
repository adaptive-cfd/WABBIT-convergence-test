#!/bin/bash
mpi="mpirun -np 8"
ini="disc-convection.ini"
bs=17

JMAX=( 2 3 4 5 6 7 8 )
EPS=( 1.0e-01 4.7e-02 1.4e-02 2.2e-03 1.3e-04 3.2e-06 5.9e-08 )
#1.00000000e-02   6.15848211e-03   3.79269019e-03
#2.33572147e-03   1.43844989e-03   8.85866790e-04
#5.45559478e-04   3.35981829e-04   2.06913808e-04
#1.27427499e-04   7.84759970e-05   4.83293024e-05
#2.97635144e-05   1.83298071e-05   1.12883789e-05
#6.95192796e-06   4.28133240e-06   2.63665090e-06
#1.62377674e-06   1.00000000e-06)

# as Jmax=4 and eps=1e-6 makes no sense, you can skip it. just the smallest 1e-7 value is contained!
pre="opt_eps_CDF44"
name="disc"

# delete all data:
rm -rf ${pre}_*


for (( a=0; a<=6; a++ ))
do

		dir=${pre}_${name}_Bs${bs}_Jmax${JMAX[$a]}_eps${EPS[$a]}

		if [ ! -f "$dir"/phi_000002500000.h5 ]; then
			mkdir $dir
			echo $dir
			cd $dir
			cp ../$ini .

			ln -s ../wabbit

			../replace_ini_value.sh $ini Discretization order_discretization FD_4th_central_optimized
			../replace_ini_value.sh $ini Discretization order_predictor multiresolution_4th
			../replace_ini_value.sh $ini Wavelet transform_type biorthogonal 

			../replace_ini_value.sh $ini Blocks adapt_mesh 1
			../replace_ini_value.sh $ini Blocks adapt_inicond 1
			../replace_ini_value.sh $ini Blocks eps ${EPS[$a]}
			../replace_ini_value.sh $ini Blocks number_block_nodes $bs
			../replace_ini_value.sh $ini Blocks number_ghost_nodes 6
			../replace_ini_value.sh $ini Blocks max_treelevel ${JMAX[$a]}
			../replace_ini_value.sh $ini Blocks min_treelevel 1

			$mpi ./wabbit $ini --memory=3.0GB
			cd ..
		else
			echo "Test already done:" $dir
		fi
done
