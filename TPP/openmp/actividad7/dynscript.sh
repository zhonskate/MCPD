#!/bin/sh
#PBS -l nodes=1,walltime=00:10:00
#PBS -q mcpd
#PBS -d .
#PBS -j oe
#PBS -o resultsdyn
cat $PBS_NODEFILE
OMP_NUM_THREADS=32 OMP_SCHEDULE=dynamic ./sudoku_task