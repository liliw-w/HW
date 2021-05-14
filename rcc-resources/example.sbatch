#!/bin/bash
#SBATCH --job-name=example_python
#SBATCH --output=script.out
#SBATCH --error=script.err
#SBATCH --time=00:05:00
#SBATCH --partition=broadwl
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=2000

module load python
python script.py
