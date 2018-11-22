echo "" > omp-$1-results.txt

for i in 1 2 4 8
do
    for j in 10 20 50 100 200 500 1000 2000 5000 10000 20000
    do
        ./openmp -n $1 -i $j -c 0 -t $i -v 2 >> omp-$1-results.txt
        echo "[N=$1] finished threads $i iterations $j"
    done
done