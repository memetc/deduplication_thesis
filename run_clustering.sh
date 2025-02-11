#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --job-name=clustering_pipeline
#SBATCH --time=4-00:00:00
#SBATCH --mem=250GB
#SBATCH --partition=compms-gpu-a40
#SBATCH --nodelist=compms-gpu-2.exbio.wzw.tum.de
#SBATCH --gres=gpu:1
#SBATCH --output=outputs/clustering_pipeline_%j.out
#SBATCH --error=outputs/clustering_pipeline_%j.err

# Load conda environment
source /cmnfs/home/students/m.celimli/notebooks/conda.sh
conda activate dev_env_mehmet

# Set PYTHONPATH to include the current working directory
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Accept a command-line argument for the Python script to run.
# If no argument is provided, default to main.py.
PY_SCRIPT=${1:-main.py}
echo "Running clustering pipeline using ${PY_SCRIPT}"

# Run the specified Python script using srun
srun python ${PY_SCRIPT}
