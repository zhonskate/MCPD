gcc -fopenmp -llapack -lblas -lm -o cholesky_escalar cholesky_escalar.c ctimer.c
gcc -fopenmp -llapack -lblas -lm -o cholesky cholesky.c ctimer.c
