#!/bin/bash

#SBATCH --partition=glados-gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=20G
#SBATCH --gres=gpu:4
#SBATCH --time=7-00:00:00
#SBATCH --array=137-230
#SBATCH --output="outslurm/slurm-%A_%a.out"

day1=`sed -n "${SLURM_ARRAY_TASK_ID}p" dates_X7-13.list`
station="G12" 

echo "station $station, start date is $day1"

module load cuda/9.1
python pyscamp_1day_BH.py $station $day1
