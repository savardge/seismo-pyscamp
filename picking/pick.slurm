#!/bin/bash

#SBATCH --partition=geo
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=3G
#SBATCH --nodes=1
#SBATCH --time=10:00:00
#SBATCH --array=1-15

xmlfile=`sed -n "${SLURM_ARRAY_TASK_ID}p" xmlfiles.list`
echo "Starting detection for file: $xmlfile"
python aggregate_scamp_detections.py $xmlfile
