#!/bin/bash

#SBATCH --partition=geo,glados12,glados16
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=3G
#SBATCH --time=7-00:00:00
#SBATCH --array=1-779
#SBATCH --output="outslurm/slurm-%A_%a.out"

mpfile=`sed -n "${SLURM_ARRAY_TASK_ID}p" allfiles.list`

python extract_detections.py $mpfile

