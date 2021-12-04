#!/bin/bash

inputfile=$1

for prefix in `cat $inputfile`; do

	nfiles=`ls matrix_profiles/$prefix*_mp.npy | wc -l`
	if [ "$nfiles" -eq 3 ]
	then
		if ! ls detections_combined/$prefix*COMBINED* &> /dev/null
		then
			ninput=`ls matrix_profiles/$prefix*_detections.csv|wc -l`
			if [ "$ninput" -gt 2 ]
			then
				echo "TODO:  $prefix"
			else
				echo "missing detection files."
				wc -l matrix_profiles/$prefix*_detections.csv
			fi
		fi
	fi

done
