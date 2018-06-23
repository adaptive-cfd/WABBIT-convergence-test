#!/bin/bash

ini="halfswirl.ini"
bs=33

EPS=( 	1.00000000e-02   6.15848211e-03   3.79269019e-03
2.33572147e-03   1.43844989e-03   8.85866790e-04
5.45559478e-04   3.35981829e-04   2.06913808e-04
1.27427499e-04   7.84759970e-05   4.83293024e-05
2.97635144e-05   1.83298071e-05   1.12883789e-05
6.95192796e-06   4.28133240e-06   2.63665090e-06
1.62377674e-06   1.00000000e-06   1.00000000e-07 )

pre="adaptive"

# do not delete prefix!!! it can delete existing data for Jmax /= INf

for eps in ${EPS[@]}
do

	dir=${pre}_halfswirl_Bs${bs}_JmaxInf_eps${eps}

	if [ ! -f "$dir"/phi_000002500000.h5 ]; then
		mkdir $dir
		echo $dir
		cd $dir
		cp ../$ini .

		ln -s ../wabbit

		../replace_ini_value.sh $ini Discretization order_discretization FD_4th_central_optimized
		../replace_ini_value.sh $ini Discretization order_predictor multiresolution_4th

		../replace_ini_value.sh $ini Blocks adapt_mesh 1
		../replace_ini_value.sh $ini Blocks adapt_inicond 1
		../replace_ini_value.sh $ini Blocks eps ${eps}
		../replace_ini_value.sh $ini Blocks number_block_nodes $bs
		../replace_ini_value.sh $ini Blocks number_ghost_nodes 4
		# jmax large
		../replace_ini_value.sh $ini Blocks max_treelevel 13
		../replace_ini_value.sh $ini Blocks min_treelevel 2

		$mpi ./wabbit 2D $ini --memory=10.0GB --new
		cd ..
	else
		echo "Test already done:" $dir
	fi
done
