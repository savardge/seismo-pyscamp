#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --partition=gpu-v100
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=100G
#SBATCH --gres=gpu:2
#SBATCH --output="outslurm/slurm-%A_MG09_DPZ.out"
###SBATCH --array=1-48
###SBATCH --output="outslurm/slurm-%A_%a.out"

#day1=`sed -n "${SLURM_ARRAY_TASK_ID}p" dates.list | awk '{print $1}'`

day1="20190710"
#day1="20190830"
station="MG09"
channel="DPZ"

module load cuda/10.0.130

numdays=27 #31 #294 # number of days MG06: 25 days between 07/07 - 08/05
sublen=10.0 #5.0  # in seconds
fs=100.0

echo "station $station, start date is $day1, no. days is $numdays, window length is $sublen s, sampling rate is $fs Hz, channel is $channel"

python call_pyscamp_1chan_ARC.py $station $day1 $numdays $sublen $channel $fs

