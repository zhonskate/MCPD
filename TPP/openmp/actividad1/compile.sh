gcc -o seq actividad1-seq.c -lm ctimer.c
gcc -o parallel actividad1-parallel.c -lm -fopenmp ctimer.c
