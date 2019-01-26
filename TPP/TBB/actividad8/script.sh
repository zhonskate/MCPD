#!/bin/sh
#PBS -l nodes=1,walltime=00:10:00
#PBS -q mcpd
#PBS -d .
#PBS -j oe
#PBS -o results.txt
cat $PBS_NODEFILE
./chhhh 100 250
./chhhh 250 250
./chhhh 500 250
./chhhh 750 250
./chhhh 1000 250
./chhhh 1250 250
./chhhh 1500 250
./chhhh 1750 250
./chhhh 2000 250
./chhhh 2250 250
./chhhh 2500 250
./chhhh 2750 250
./chhhh 3000 250
./chhhh 3250 250
./chhhh 3500 250
./chhhh 3750 250
./chhhh 4000 250
./chhhh 4250 250
./chhhh 4500 250