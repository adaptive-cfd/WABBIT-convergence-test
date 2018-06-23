#!/bin/bash

ini="halfswirl.ini"
# we fix the upper limit to 2048 points
bs=33
minlevel=2
maxlevel=6


# resolution (in points) is N = (Bs-1)*2^Jmax

pre=equidistant


# delete all data:
rm -rf ${pre}_*


for (( j=minlevel; j<=maxlevel; j++ ))
do
	dir=${pre}_halfswirl_Bs${bs}_Jmax${j}

	if [ ! -f "$dir"/phi_000002500000.h5 ]; then
		mkdir $dir
		echo $dir
		cd $dir
		cp ../$ini .

		ln -s ../wabbit

		../replace_ini_value.sh $ini Discretization order_discretization FD_4th_central_optimized
		../replace_ini_value.sh $ini Discretization order_predictor multiresolution_4th

		../replace_ini_value.sh $ini Blocks adapt_mesh 0
		../replace_ini_value.sh $ini Blocks adapt_inicond 0
		../replace_ini_value.sh $ini Blocks number_block_nodes $bs
		../replace_ini_value.sh $ini Blocks number_ghost_nodes 4
		../replace_ini_value.sh $ini Blocks max_treelevel $j
		../replace_ini_value.sh $ini Blocks min_treelevel $j

		$mpi ./wabbit 2D $ini --memory=10.0GB --new
		cd ..
	else
		echo "Test already done:" $dir
	fi
done
