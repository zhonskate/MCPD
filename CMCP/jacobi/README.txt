OpenMP:

    - Compilación:
        $ gcc -o jacobi_omp jacobi_omp.c -fopenmp -lm
    
    - Ejecución:
        $ ./jacobi_omp -h (muestra la ayuda).

MPI

    - Compilación:
        $ mpicc -o jacobi_mpi jacobi_mpi.c
    
    - Ejecución:
        $ mpirun -c <Hilos> jacobi_mpi <Iteraciones> <verbose> <N (orden de la matriz))>

        p.ej: mpirun -c 2 jacobi_mpi 2000 1 4