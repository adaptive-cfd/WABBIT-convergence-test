#!/bin/bash

# Pretty colors
# Regular Colors
Black='\e[0;30m'        # Black
Red='\e[0;31m'          # Red
Green='\e[0;32m'        # Green
Yellow='\e[0;33m'       # Yellow
Blue='\e[0;34m'         # Blue
Purple='\e[0;35m'       # Purple
Cyan='\e[0;36m'         # Cyan
White='\e[0;37m'        # White
Color_Off='\e[0m'       # Text Reset

for dir in *_halfswirl_Bs*
do
    echo $dir
    cd $dir

    #rm fullphi_000002500000.h5


     if [ ! -f "fullphi_000002500000.h5" ]; then

    if [ -f "phi_000002500000.h5" ]; then

        echo $dir/"phi_000002500000.h5"
        
        bs=$(cat halfswirl.ini | grep number_block_nodes)
        bs=${bs##"number_block_nodes="}
        bs=${bs%%";"}
        
        level=7
        if [ "${bs}" == 17 ]; then
            # 2048 points
            level=7
        fi
        if [ "${bs}" == 33 ]; then
            # 2048 points        
            level=6
        fi
        if [ "${bs}" == 65 ]; then
            # 2048 points
            level=5
        fi

        yes=1
        counter=0
        while [ ! "$yes" == "0" ]
        do
            $mpi ../../wabbit-post 2D --sparse-to-dense phi_000002500000.h5 fullphi_000002500000.h5 $level 4 --new
            yes="$?"

            if [ "$yes" == "0" ]; then
                echo $dir/"phi_000002500000.h5"
                echo -e $Green"Cool, that did work!"$Color_Off
            else
                echo $dir/"phi_000002500000.h5"
                
                counter=$((counter+1))
                echo -e $Red"it appears something went wrong.." $counter$Color_Off
                if [ "$counter" == 10 ]; then
                    echo -e $Red"fuck it i give up"$Color_Off
                    yes=0
                fi
                sleep 3
            fi
        done

    fi
     fi

    cd ..
done
