#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --job-name=clustering_pipeline
#SBATCH --time=4-00:00:00
#SBATCH --mem=50GB
#SBATCH --partition=compms-gpu-a40
#SBATCH --nodelist=compms-gpu-1.exbio.wzw.tum.de
#SBATCH --gres=gpu:1

# Load conda environment
source /cmnfs/home/students/m.celimli/notebooks/conda.sh
conda activate dev_env_mehmet

# Set environment variables
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Run the clustering pipeline
srun python main.py
