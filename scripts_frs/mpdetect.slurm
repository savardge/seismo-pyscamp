#!/bin/bash

#SBATCH --partition=geo,glados12,glados16
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=20G
#SBATCH --time=1:00:00
#SBATCH --array=1-75
#SBATCH --output="outslurm/slurm-%A_%a.out"


# STEP 1: EXTRACT detections from matrix profile (for 1 channel)

#mpfile=`sed -n "${SLURM_ARRAY_TASK_ID}p" mpfiles.list`
#python extract_detections2.py $mpfile


# STEP 2: COMBINE 3 channels

#prefix=`sed -n "${SLURM_ARRAY_TASK_ID}p" prefixes.list`
#chan2="DP"
#suffix="_250Hz_win125samp_detections.csv"
#python combine.py $prefix $suffix $chan2


# STEP 4: GROUP detections by linking

csvfile=`sed -n "${SLURM_ARRAY_TASK_ID}p" inputcsv.list`
python group.py $csvfile
