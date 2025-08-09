#!/bin/bash
#SBATCH --job-name=distilroberta_training
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --output=output.out
#SBATCH --error=error.err
#SBATCH --partition=data-science
#SBATCH --account=pregrado
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hector.nieto@utec.edu.pe
srun python3 main.py