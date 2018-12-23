#!/bin/sh
#PBS -l nodes=1,walltime=00:10:00
#PBS -q mcpd
#PBS -d .
#PBS -j oe
#PBS -o 1n
cat $PBS_NODEFILE
OMP_NUM_THREADS=1 OMP_SCHEDULE=dynamic ./cholesky 1000 250
OMP_NUM_THREADS=1 OMP_SCHEDULE=dynamic ./cholesky 2000 250
OMP_NUM_THREADS=1 OMP_SCHEDULE=dynamic ./cholesky 3000 250
OMP_NUM_THREADS=1 OMP_SCHEDULE=dynamic ./cholesky 4000 250
OMP_NUM_THREADS=1 OMP_SCHEDULE=dynamic ./cholesky 5000 250
