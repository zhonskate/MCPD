#!/bin/bash

mv $1-matrix-data-jacobi.inp matrix-data-jacobi.inp
mv $1-vector-data-jacobi.inp vector-data-jacobi.inp

echo "" > mpi-$1-results.txt

for i in 1 2 4 8
do
    for j in 10 20 50 100 200 500 1000 2000 5000 10000 20000
    do
        mpirun -c $i mpi $j 2 >> mpi-$1-results.txt
        echo "[N=$1] finished threads $i iterations $j"
    done
done

mv matrix-data-jacobi.inp $1-matrix-data-jacobi.inp
mv vector-data-jacobi.inp $1-vector-data-jacobi.inp