#!/bin/sh
#PBS -l nodes=1,walltime=00:10:00
#PBS -q mcpd
#PBS -d .
#PBS -j oe
#PBS -o 4salidafinal
cat $PBS_NODEFILE
OMP_NUM_THREADS=4 OMP_SCHEDULE=static ./cholesky 100 250
OMP_NUM_THREADS=4 OMP_SCHEDULE=static ./cholesky 250 250
OMP_NUM_THREADS=4 OMP_SCHEDULE=static ./cholesky 500 250
OMP_NUM_THREADS=4 OMP_SCHEDULE=static ./cholesky 750 250
OMP_NUM_THREADS=4 OMP_SCHEDULE=static ./cholesky 1000 250
OMP_NUM_THREADS=4 OMP_SCHEDULE=static ./cholesky 1250 250
OMP_NUM_THREADS=4 OMP_SCHEDULE=static ./cholesky 1500 250
OMP_NUM_THREADS=4 OMP_SCHEDULE=static ./cholesky 1750 250
OMP_NUM_THREADS=4 OMP_SCHEDULE=static ./cholesky 2000 250
OMP_NUM_THREADS=4 OMP_SCHEDULE=static ./cholesky 2250 250
OMP_NUM_THREADS=4 OMP_SCHEDULE=static ./cholesky 2500 250
OMP_NUM_THREADS=4 OMP_SCHEDULE=static ./cholesky 2750 250
OMP_NUM_THREADS=4 OMP_SCHEDULE=static ./cholesky 3000 250
OMP_NUM_THREADS=4 OMP_SCHEDULE=static ./cholesky 3250 250
OMP_NUM_THREADS=4 OMP_SCHEDULE=static ./cholesky 3500 250
OMP_NUM_THREADS=4 OMP_SCHEDULE=static ./cholesky 3750 250
OMP_NUM_THREADS=4 OMP_SCHEDULE=static ./cholesky 4000 250
OMP_NUM_THREADS=4 OMP_SCHEDULE=static ./cholesky 4250 250
OMP_NUM_THREADS=4 OMP_SCHEDULE=static ./cholesky 4500 250
