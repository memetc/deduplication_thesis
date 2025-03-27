#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --job-name=apply_cl
#SBATCH --time=4-00:00:00
#SBATCH --mem=50GB
#SBATCH --cpus-per-task=8
#SBATCH --partition=shared-cpu
#SBATCH --output=outputs/apply_cl%j.out
#SBATCH --error=outputs/apply_cl%j.err

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
srun python apply_cluster_labels.py
