#!/bin/sh
#PBS -l nodes=1,walltime=00:10:00
#PBS -q mcpd
#PBS -d .
#PBS -j oe
#PBS -o salida
cat $PBS_NODEFILE
./seq 40000 40000
