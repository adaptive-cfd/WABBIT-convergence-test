#!/bin/bash

ini="halfswirl.ini"
# we fix the upper limit to 1024 points
blocksize=( 17 33 65 )
minlevel=( 2 2 2 )
maxlevel=( 6 5 4 )
# resolution (in points) is N = (Bs-1)*2^Jmax

pre=equidistant


# delete all data:
rm -rf ${pre}_*


for (( a=0; a<=2; a++ ))
do

	bs=${blocksize[a]}
	for (( j=${minlevel[$a]}; j<=${maxlevel[$a]}; j++ ))
	do
		dir=${pre}_halfswirl_Bs${bs}_Jmax${j}
		mkdir $dir
		echo $dir
		cd $dir
		cp ../$ini .

		ln -s ../wabbit

		../replace_ini_value.sh $ini Discretization order_discretization FD_4th_central_optimized
		../replace_ini_value.sh $ini Discretization order_predictor multiresolution_4th

		../replace_ini_value.sh $ini Blocks adapt_mesh 0
		../replace_ini_value.sh $ini Blocks adapt_inicond 0
		# ../replace_ini_value.sh $ini Blocks eps
		../replace_ini_value.sh $ini Blocks number_block_nodes $bs
		../replace_ini_value.sh $ini Blocks number_ghost_nodes 4
		../replace_ini_value.sh $ini Blocks max_treelevel $j
		../replace_ini_value.sh $ini Blocks min_treelevel $j

		echo $mpi ./wabbit 2D $ini --memory=5.0GB
		cd ..
	done
done
