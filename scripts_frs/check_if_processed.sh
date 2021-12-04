#!/bin/bash

inputfile=$1
sta="BH012"
wfdir="/home/gilbert_lab/cami_frs/borehole_data/sac_daily_nez_500Hz"

for day in `cat $inputfile`; do

	if [ -d "$wfdir/$day" ]; then

		for chan in DPN DPE DPZ; do
#			echo $day $chan

			if ls $wfdir/$day/*$sta*$chan* &> /dev/null
			then
				if ! ls  matrix_profiles/$day*$day*$sta*$chan*_mp.npy &> /dev/null
				then
#					echo "NOT processed"
#					ls matrix_profiles/$day*$chan*_mp.npy
					echo $day $chan
#				else
#					echo "Processed"
#					ls matrix_profiles/$day*$chan*_mp.npy
				fi
			fi
		done
	fi
done
