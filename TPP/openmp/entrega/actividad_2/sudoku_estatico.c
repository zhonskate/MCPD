#include <stdio.h>
#include <stdlib.h>
#include "ctimer.h"
#include "sudoku.h"
#include <omp.h>

int main( int argc, char *argv[] ) {
  int sol[81];

  if( argc < 2 ) {
    printf("Usage: %s profundidad\n",argv[0]);
    exit(1);
  }
  int profundidad;
  sscanf(argv[1],"%d",&profundidad);
  printf("sudoku inicial: \n");
  //init_sudoku("normal",sol);
  init_sudoku("muy dificil",sol);
  prin_sudoku(sol);

  double tiempo, ucpu, scpu;
  ctimer( &tiempo, &ucpu, &scpu );

  int A[3000][81];
  int B[3000][81];
  int nivel, nodo, k, l;
  for(int l = 0; l < 81; l++) A[0][l] = sol[l]; 
  int tableros = 1;
  for( int nivel = 0; nivel < profundidad; nivel ++){
    int j = 0;
    for( int nodo = 0; nodo < tableros; nodo++ ) {
      int k = 0; while( k < 81 && A[nodo][k] != 0 ) k++;
      if( k<81 ) {
        for( int i=1; i<=9; i++ ) {
          A[nodo][k] = i;
          if( es_factible( k/9+1, k%9+1, A[nodo] ) ) {
            for( int l = 0; l<81; l++ ) { B[j][l] = A[nodo][l]; }
            j++;
          } 
          A[nodo][k] = 0;
        } 
      } 
    } 
    tableros = j;
    for( int i = 0; i<tableros; i++ )
      for( int k = 0; k<81; k++ )
        A[i][k] = B[i][k];
  }
  printf("Tableros = %d\n",tableros);

  #pragma omp parallel for schedule(runtime)
  for(int tablero = 0; tablero < tableros; tablero++) {
    int mascara[81];
    for ( int i = 0; i < 81; i++ ) mascara[i] = A[tablero][i] != 0;
    sudoku_sol(1,1,A[tablero],mascara);
  }

  ctimer( &tiempo, &ucpu, &scpu );
  printf("profundidad: %d Tiempo = %f\n",profundidad,tiempo);

  return 0;
}

void sudoku_sol( int i, int j, int sol[81], int mascara[81] ) {
   int k;
   if( mascara(i, j) == 0 ) {
      for( k = 1; k <= 9; k++ ) {
         sol( i, j ) = k;                                
         if( es_factible( i, j, sol ) ) {
            if( i == 9 && j == 9 ) {
               printf("Solucion: \n");
               prin_sudoku(sol);
            }
            if( i < 9 && j == 9 ) {
               sudoku_sol ( i+1, 1, sol, mascara );
            }
            if( i <= 9 && j < 9 ) {
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
         sudoku_sol ( i+1, 1, sol, mascara );
      }
      if( i <= 9 && j < 9 ) {
         sudoku_sol( i , j+1, sol, mascara );
      }
   }
}

