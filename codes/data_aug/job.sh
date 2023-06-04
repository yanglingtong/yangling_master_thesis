#!/bin/bash

##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=JobExample1      #Set the job name to "JobExample1"
#SBATCH --time=00:30:00              #Set the wall clock limit to 1hr and 30min
#SBATCH --nodes=1
#SBATCH --ntasks=1                 #Request 1 task
#SBATCH --mem=20000M                  #Request 2560MB (2.5GB) per node
#SBATCH --output=Example1Out.%j    #Send stdout/err to "Example1Out.[jobID]"
#SBATCH --gres=gpu:1                 #Request 2 GPU per node can be 1 or 2
#SBATCH --partition=gpu              #Request the GPU partition/queue

##OPTIONAL JOB SPECIFICATIONS
##SBATCH --account=123456             #Set billing account to 123456
##SBATCH --mail-type=ALL              #Send email on all job events    #Send all emails to email_address

#First Executable Line
#python a_model_selection_balance_4.py
python experiment_1.py
#python create_dataset.py