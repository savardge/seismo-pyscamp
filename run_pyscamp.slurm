#!/bin/bash

#SBATCH --partition=glados-gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=40G
#SBATCH --gres=gpu:4
#SBATCH --time=7-00:00:00
#SBATCH --array=20 #36
#SBATCH --output="outslurm/slurm-%A_%a.out"

day1=`sed -n "${SLURM_ARRAY_TASK_ID}p" dates.list`
station="G12"

echo "station $station, start date is $day1"

module load cuda/9.1

numdays=14 # number of days
sublen=30.0  # in seconds
python call_pyscamp.py $station $day1 $numdays $sublen
