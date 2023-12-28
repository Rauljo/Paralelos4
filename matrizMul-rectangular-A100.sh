#!/bin/sh
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --mem 64G
#SBATCH --gres gpu:a100
#SBATCH -t 00:30:00
for tam in 250 500 1000 1500 2000 2500 3000 3500 4000 4500 5000; do
    for threads in 1 2 4 8 16 32; do
        srun matriz_rectangular ${tam} ${tam} ${tam} ${tam} ${threads} ${threads}
    done
done
