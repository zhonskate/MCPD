#!/bin/sh
#PBS -l nodes=1,walltime=00:10:00
#PBS -q mcpd
#PBS -d .
#PBS -j oe
#PBS -o resultsseq
cat $PBS_NODEFILE
./act4 40000 40000 
