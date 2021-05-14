#!/bin/bash
#SBATCH --job-name=mc_integration
#SBATCH --output=mc_int_%j.out
#SBATCH --error=mc_int_%j.err
#SBATCH --time=00:05:00
#SBATCH --partition=broadwl
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=2000
#SBATCH --account=caam37830

module load python
conda activate pycourse

for i in `seq 1 4`; do
  srun \
    -n 1 \
    python mc_integration.py 1000000 --file mc_node_$i.csv
done


