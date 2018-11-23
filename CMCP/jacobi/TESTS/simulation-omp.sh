!/bin/bash

for i in 2 4 8 16 24 32 40 56 80 112 160 224 320 456 640 904 1280 1808 2560
do
    ./run_test_openmp.sh $i
    echo "finished $i"
done
