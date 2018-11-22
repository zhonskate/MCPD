#!/bin/bash

echo "" > seq-$1-results.txt


for j in 2000 
do
    ./openmp -n $1 -i $j -c 0 -t 1 -v 3 >> seq-$1-results.txt
    echo "[N=$1] finished sequential iterations $j"
done