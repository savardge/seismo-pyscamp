#!/bin/bash

#SBATCH --partition=gpu-v100
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=40G
#SBATCH --gres=gpu:2
#SBATCH --time=24:00:00
###SBATCH --array=50-302:7
#SBATCH --array=1-637
#SBATCH --output="outslurm/slurm-%A_%a.out"

day1=`sed -n "${SLURM_ARRAY_TASK_ID}p" days_to_redo.list | awk '{print $1}'`
channel=`sed -n "${SLURM_ARRAY_TASK_ID}p" days_to_redo.list | awk '{print $2}'`
station="G12"

echo "station $station, start date is $day1, channel is $channel"

module load cuda/10.0.130

numdays=1 # number of days
sublen=0.5  # in seconds
python call_pyscamp_1chan.py $station $day1 $numdays $sublen $channel
#python call_pyscamp.py $station $day1 $numdays $sublen
