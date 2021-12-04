#!/bin/bash
#SBATCH --time=2:00:00
#SBATCH --partition=glados-gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=20G
#SBATCH --gres=gpu:3
#SBATCH --array=1-267
#SBATCH --output="outslurm/slurm-%A_%a.out"

INFILE="input_files/stachan.list"
day1=`sed -n "${SLURM_ARRAY_TASK_ID}p" $INFILE | awk '{print $1}'`
channel=`sed -n "${SLURM_ARRAY_TASK_ID}p" $INFILE | awk '{print $2}'`
station="BH012"

echo "station $station, start date is $day1, channel is $channel"

module load cuda/9.1

numdays=1 #31 #294 # number of days
sublen=0.5 #5.0  # in seconds
fs=250.0
python python_scripts/step1_pyscamp_1chan.py $station $day1 $numdays $sublen $channel $fs

