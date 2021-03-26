#!/bin/bash
#SBATCH --time=7-00:00:00
#SBATCH --partition=glados-gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=100G
#SBATCH --gres=gpu:4
#SBATCH --time=7-00:00:00
#SBATCH --output="outslurm/slurm-%A_RDG_HHX.out"
###SBATCH --array=1-48
###SBATCH --output="outslurm/slurm-%A_%a.out"

#day1=`sed -n "${SLURM_ARRAY_TASK_ID}p" dates.list | awk '{print $1}'`

day1="20190829"
station="RDG"
channel="HHX"

module load cuda/9.1

numdays=59 #31 #294 # number of days
sublen=10.0 #5.0  # in seconds
fs=100.0

echo "station $station, start date is $day1, no. days is $numdays, window length is $sublen s, sampling rate is $fs Hz, channel is $channel"

python call_pyscamp_1chan.py $station $day1 $numdays $sublen $channel $fs

