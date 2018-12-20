gcc -std=gnu99 -o sudoku sudoku.c libsudoku.o ctimer.c init_sudoku.c
gcc -std=gnu99 -fopenmp -o sudoku_task sudoku_task.c libsudoku.o ctimer.c init_sudoku.c
