#!/bin/bash

for i in 2 4 8 16 24 32 40 56 80 112 160 224 320 456 640 904 1280 1808 2560 3624 5120 7240 10240
do
    ./run_test_seq.sh $i
    echo "finished $i"
done