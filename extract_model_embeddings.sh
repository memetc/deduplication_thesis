#!/bin/bash
 
#SBATCH --ntasks=1
#SBATCH --job-name=extract_model_embeddings
#SBATCH --time=4-00:00:00
#SBATCH --mem=200GB
#SBATCH --partition=shared-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --output=outputs/extract_model_embeddings%j.out
#SBATCH --error=outputs/extract_model_embeddings%j.err

source /cmnfs/home/students/m.celimli/notebooks/conda.sh
conda activate dev_env_mehmet

export PYTHONPATH="${PYTHONPATH}:$(pwd)"

srun python get_embeddings_from_model.py


