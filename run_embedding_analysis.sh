#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --job-name=embedding_analysis
#SBATCH --time=4-00:00:00
#SBATCH --mem=5GB
#SBATCH --cpus-per-task=8
#SBATCH --partition=compms-cpu-big
#SBATCH --output=outputs/embedding_analysis-%j.out
#SBATCH --error=outputs/embedding_analysis-%j.err

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
srun python embedding_analysis.py
