OpenMP:

    - Compilación:
        $ gcc -o jacobi_omp jacobi_omp.c -fopenmp -lm
    
    - Ejecución:
        $ ./jacobi_omp -h (muestra la ayuda).
    
    //Verbose = 0 -> no hay output
    //Verbose = 1 -> todo el output
    //Verbose = 2 -> output paralelo
    //Verbose = 3 -> output solo secuencial

MPI

    - Compilación:
        $ mpicc -o jacobi_mpi jacobi_mpi.c
    
    - Ejecución:
        $ mpirun -c <Hilos> jacobi_mpi <Iteraciones> <verbose> <N (orden de la matriz))>

        p.ej: mpirun -c 2 jacobi_mpi 2000 1 4
    
    //Verbose = 0 -> no hay output
    //Verbose = 1 -> todo el output
    //Verbose = 2 -> output paralelo