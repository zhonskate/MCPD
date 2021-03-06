#!/bin/sh
#PBS -l nodes=1,walltime=00:10:00
#PBS -q mcpd
#PBS -d .
#PBS -j oe
#PBS -o resultsstat
cat $PBS_NODEFILE
OMP_NUM_THREADS=1 OMP_SCHEDULE=static ./sudoku_task
OMP_NUM_THREADS=2 OMP_SCHEDULE=static ./sudoku_task
OMP_NUM_THREADS=4 OMP_SCHEDULE=static ./sudoku_task
OMP_NUM_THREADS=8 OMP_SCHEDULE=static ./sudoku_task
OMP_NUM_THREADS=16 OMP_SCHEDULE=static ./sudoku_task
OMP_NUM_THREADS=32 OMP_SCHEDULE=static ./sudoku_task
