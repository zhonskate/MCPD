#!/bin/sh
#PBS -l nodes=4,walltime=00:10:00
#PBS -q cpa
#PBS -d .
cat $PBS_NODEFILE


mpirun -np 4 ./lsqr128 -fileA /scratch/mnichipr/A128_30.petsc -fileG /scratch/mnichipr/b128_30_forbild.petsc  -svd_nsv 16384
