#!/bin/bash

# note bizarre numbering..
test3=1 # easy, 4th-4th-3rd, equidistant
test5=1 #
test3=1
test2=1

#---------------------------------------------------------------------------------------------------

if [ "$test3" == 1 ]; then
	ini=swirl_RK3TVD.ini
	dt=(2.5e-3 1.0e-3 5.0e-4 2.5e-4 1.0e-4)
	test_number=3
	test_name="equi_4th-4th-3rd"

	# delete all data:
	rm -rf dt${test_number}_*
	# run tests
	for ddt in ${dt[@]}
	do
		dir=dt${test_number}_${test_name}_${ddt}
		mkdir $dir
		cd $dir
		cp ../$ini .
		ln -s ../wabbit

		# time
		../replace_ini_value.sh $ini Time time_max 4.0
		../replace_ini_value.sh $ini Time dt_fixed $ddt
		../replace_ini_value.sh $ini Time CFL 9.9

		# order
		../replace_ini_value.sh $ini Discretization order_discretization FD_4th_central_optimized
		../replace_ini_value.sh $ini Discretization order_predictor multiresolution_4th

		# blocks
		../replace_ini_value.sh $ini Blocks adapt_mesh 0
		../replace_ini_value.sh $ini Blocks adapt_inicond 0
		../replace_ini_value.sh $ini Blocks inicond_refinements 0
		../replace_ini_value.sh $ini Blocks number_block_nodes 17
		../replace_ini_value.sh $ini Blocks number_ghost_nodes 4
		../replace_ini_value.sh $ini Blocks eps 1.0e-4
		../replace_ini_value.sh $ini Blocks max_treelevel 12
		../replace_ini_value.sh $ini Blocks min_treelevel 2

		# other
		../replace_ini_value.sh $ini ConvectionDiffusion nu 0.0
		../replace_ini_value.sh $ini ConvectionDiffusion blob_width 0.01

		$mpi ./wabbit 2D $ini --memory=3.0GB
		cd ..
	done
fi

#---------------------------------------------------------------------------------------------------

if [ "$test5" == 1 ]; then
	ini=swirl-nonequi-nonadaptive.ini
	dt=(5.0e-3 4.0e-3 3.0e-3 2.5e-3 1.0e-3 5.0e-4)
	test_number=5
	test_name="nonequi_4th-4th-4th"

	# delete all data:
	rm -rf dt${test_number}_*
	# run tests
	for ddt in ${dt[@]}
	do
		dir=dt${test_number}_${test_name}_${ddt}
		mkdir $dir
		cd $dir
		cp ../$ini .
		ln -s ../wabbit

		# time
		../replace_ini_value.sh $ini Time time_max 4.0
		../replace_ini_value.sh $ini Time dt_fixed $ddt
		../replace_ini_value.sh $ini Time CFL 9.9

		# order
		../replace_ini_value.sh $ini Discretization order_discretization FD_4th_central_optimized
		../replace_ini_value.sh $ini Discretization order_predictor multiresolution_4th

		# blocks
		../replace_ini_value.sh $ini Blocks adapt_mesh 0
		../replace_ini_value.sh $ini Blocks adapt_inicond 1
		../replace_ini_value.sh $ini Blocks inicond_refinements 1
		../replace_ini_value.sh $ini Blocks number_block_nodes 17
		../replace_ini_value.sh $ini Blocks number_ghost_nodes 4
		../replace_ini_value.sh $ini Blocks eps 1.0e-4
		../replace_ini_value.sh $ini Blocks max_treelevel 12
		../replace_ini_value.sh $ini Blocks min_treelevel 2

		# other
		../replace_ini_value.sh $ini ConvectionDiffusion nu 0.0
		../replace_ini_value.sh $ini ConvectionDiffusion blob_width 0.01

		$mpi ./wabbit 2D $ini --memory=3.0GB
		cd ..
	done
fi

#---------------------------------------------------------------------------------------------------

if [ "$test1" == 1 ]; then
	ini=swirl.ini
	dt=(7.5e-3 5.0e-3 2.5e-3 1.0e-3 5.0e-4)
	test_number=1
	test_name="equi_4th-4th-4th"

	# delete all data:
	rm -rf dt${test_number}_*
	# run tests
	for ddt in ${dt[@]}
	do
		dir=dt${test_number}_${test_name}_${ddt}
		mkdir $dir
		cd $dir
		cp ../$ini .
		ln -s ../wabbit

		# time
		../replace_ini_value.sh $ini Time time_max 4.0
		../replace_ini_value.sh $ini Time dt_fixed $ddt
		../replace_ini_value.sh $ini Time CFL 9.9

		# order
		../replace_ini_value.sh $ini Discretization order_discretization FD_4th_central_optimized
		../replace_ini_value.sh $ini Discretization order_predictor multiresolution_4th

		# blocks
		../replace_ini_value.sh $ini Blocks adapt_mesh 0
		../replace_ini_value.sh $ini Blocks adapt_inicond 0
		../replace_ini_value.sh $ini Blocks inicond_refinements 0
		../replace_ini_value.sh $ini Blocks number_block_nodes 17
		../replace_ini_value.sh $ini Blocks number_ghost_nodes 4
		../replace_ini_value.sh $ini Blocks eps 1.0e-4
		../replace_ini_value.sh $ini Blocks max_treelevel 12
		../replace_ini_value.sh $ini Blocks min_treelevel 2

		# other
		../replace_ini_value.sh $ini ConvectionDiffusion nu 0.0
		../replace_ini_value.sh $ini ConvectionDiffusion blob_width 0.01

		$mpi ./wabbit 2D $ini --memory=3.0GB
		cd ..
	done
fi

#---------------------------------------------------------------------------------------------------
if [ "$test2" == 1 ]; then
	ini=swirl.ini
	dt=(7.5e-3 5.0e-3 2.5e-3 1.0e-3 5.0e-4)
	test_number=2
	test_name="equi_2nd-2nd-4th"

	# delete all data:
	rm -rf dt${test_number}_*
	# run tests
	for ddt in ${dt[@]}
	do
		dir=dt${test_number}_${test_name}_${ddt}
		mkdir $dir
		cd $dir
		cp ../$ini .
		ln -s ../wabbit

		# time
		../replace_ini_value.sh $ini Time time_max 4.0
		../replace_ini_value.sh $ini Time dt_fixed $ddt
		../replace_ini_value.sh $ini Time CFL 9.9

		# order
		../replace_ini_value.sh $ini Discretization order_discretization FD_2nd_central
		../replace_ini_value.sh $ini Discretization order_predictor multiresolution_2nd

		# blocks
		../replace_ini_value.sh $ini Blocks adapt_mesh 0
		../replace_ini_value.sh $ini Blocks adapt_inicond 0
		../replace_ini_value.sh $ini Blocks inicond_refinements 0
		../replace_ini_value.sh $ini Blocks number_block_nodes 17
		../replace_ini_value.sh $ini Blocks number_ghost_nodes 4
		../replace_ini_value.sh $ini Blocks eps 1.0e-4
		../replace_ini_value.sh $ini Blocks max_treelevel 12
		../replace_ini_value.sh $ini Blocks min_treelevel 2

		# other
		../replace_ini_value.sh $ini ConvectionDiffusion nu 0.0
		../replace_ini_value.sh $ini ConvectionDiffusion blob_width 0.01

		$mpi ./wabbit 2D $ini --memory=3.0GB
		cd ..
	done
fi
