#!/bin/sh
#PBS -l nodes=4,walltime=00:10:00
#PBS -q cpa
#PBS -d .
cat $PBS_NODEFILE


mpirun -np 4 ./svd -fileA /scratch/mnichipr/A32_30.petsc -fileG /scratch/mnichipr/b32_30_forbild.petsc
