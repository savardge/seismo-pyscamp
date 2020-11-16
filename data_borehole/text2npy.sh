#!/bin/bash

#SBATCH --partition=glados12,glados16,geo
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=3G
#SBATCH --time=1:00:00
#SBATCH --array=1-745
#SBATCH --output="outslurm/slurm-%A_%a.out"

fname=`sed -n "${SLURM_ARRAY_TASK_ID}p" files.list`

python text2npy.py $fname