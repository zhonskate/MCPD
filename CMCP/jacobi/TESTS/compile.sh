gcc -o openmp jacobi.c -fopenmp -std=gnu99 -lm
mpicc -o mpi mpack.c