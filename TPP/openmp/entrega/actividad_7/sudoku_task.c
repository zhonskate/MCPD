#include <stdio.h>
#include "sudoku.h"
#include "ctimer.h"
#include <omp.h>

int main( int argc, char *argv[] ) {
  int mascara[81];
  int sol[81];
  int i, j;

  printf("sudoku inicial: \n");
  init_sudoku("muy dificil",sol);
  prin_sudoku(sol);

  double tiempo, ucpu, scpu;
  ctimer( &tiempo, &ucpu, &scpu );

  for( i = 1; i <= 9; i++ ) {
    for( j = 1; j <= 9; j++ ) {
       mascara(i,j) = sol(i,j) != 0;
    }
  }
  #pragma omp parallel
  #pragma omp single
  sudoku_sol( 1, 1, sol, mascara );


  ctimer( &tiempo, &ucpu, &scpu );
  printf("Tiempo = %f\n",tiempo);

  return 0;
}

void sudoku_sol( int i, int j, int sol1[81], int mascara[81] ) {
   int k;
   int sol[81];
   for (k=0;k<81;k++) sol[k] =sol1[k];
   if( mascara(i, j) == 0 ) {
      for( k = 1; k <= 9; k++ ) {
         sol( i, j ) = k;                                
         if( es_factible( i, j, sol ) ) {
            if( i == 9 && j == 9 ) {
               printf("Solucion: \n");
               prin_sudoku(sol);
            }
            if( i < 9 && j == 9 ) {
               #pragma omp task
               sudoku_sol ( i+1, 1, sol, mascara );
            }
            if( i <= 9 && j < 9 ) {
               #pragma omp task
               sudoku_sol( i, j+1, sol, mascara );
            }
         }
      } 
      sol(i, j) = 0;                             
   } else { 
      if( i == 9 && j == 9 ) {
         printf("Solucion: \n");
         prin_sudoku(sol);
      }
      if( i < 9 && j == 9 ) {
         #pragma omp task
         sudoku_sol ( i+1, 1, sol, mascara );
      }
      if( i <= 9 && j < 9 ) {
         #pragma omp task
         sudoku_sol( i , j+1, sol, mascara );
      }
   }
}

