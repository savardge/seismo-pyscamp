#!/bin/bash
#SBATCH --time=7-00:00:00
#SBATCH --partition=glados-gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=40G
#SBATCH --gres=gpu:4
#SBATCH --time=7-00:00:00
#SBATCH --array=1-33
#SBATCH --output="outslurm/slurm-%A_%a.out"

day1=`sed -n "${SLURM_ARRAY_TASK_ID}p" dates.list | awk '{print $1}'`
channel="DPZ"
station="BH012"

#echo "station $station, start date is $day1"
echo "station $station, start date is $day1, channel is $channel"

module load cuda/9.1

numdays=1 # number of days
sublen=0.5  # in seconds
python call_pyscamp_1chan.py $station $day1 $numdays $sublen $channel
#python call_pyscamp.py $station $day1 $numdays $sublen

