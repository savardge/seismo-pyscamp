#!/bin/bash
#SBATCH --time=7-00:00:00
#SBATCH --partition=glados-gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=80G
#SBATCH --gres=gpu:4
#SBATCH --time=7-00:00:00
###SBATCH --output="outslurm/slurm-%A_5s_Dec2019.out"
#SBATCH --array=1-48
#SBATCH --output="outslurm/slurm-%A_%a.out"

day1=`sed -n "${SLURM_ARRAY_TASK_ID}p" dates.list | awk '{print $1}'`
#day1="20191201"
channel="DPZ"
station="BH012"

#echo "station $station, start date is $day1"
echo "station $station, start date is $day1, channel is $channel"

module load cuda/9.1

numdays=1 #31 #294 # number of days
sublen=0.5 #5.0  # in seconds
fs=250.0
python call_pyscamp_1chan.py $station $day1 $numdays $sublen $channel $fs

