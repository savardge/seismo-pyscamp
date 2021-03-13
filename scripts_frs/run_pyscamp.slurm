#!/bin/bash
#SBATCH --time=7-00:00:00
#SBATCH --partition=glados-gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=80G
#SBATCH --gres=gpu:4
#SBATCH --time=7-00:00:00
#SBATCH --output="outslurm/slurm-%A_Jan2019.out"

day1=`sed -n "${SLURM_ARRAY_TASK_ID}p" dates.list | awk '{print $1}'`
day1="20190101"
channel="HHZ"
station="FRS4"

#echo "station $station, start date is $day1"
echo "station $station, start date is $day1, channel is $channel"

module load cuda/9.1

numdays=31 #294 # number of days
sublen=10.0  # in seconds
fs=50.0
python call_pyscamp_1chan.py $station $day1 $numdays $sublen $channel $fs

