#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define	R(i,j)	R[j*lda+i]
#define	b(i)	b[i]
#define	x(i)	x[i]
#define	I(i)	I[i]
#define	sol(i)	sol[i]

void printMatrix( const char *s, double *R, int lda, int n );
void algoritmo1( int n, double *x, int t, double *I, double *R, int lda, double *b, double nrm );

double minimo;
double *sol;
#ifdef CONTAR_TAREAS
int tareas=0;
#endif

int main( int argc, char *argv[] ) {
  int n, N, i, j, t;
  double *R, *I, *b, *x;

  if( argc<3 ) {
    fprintf(stderr,"usage: %s n(tamano de la matriz) t(tamano del conjunto discreto) \n",argv[0]);
    exit(-1);
  }
  sscanf(argv[1],"%d",&n);
  /* Reservamos espacio para una matriz cuadrada */
  int lda = n;
  N = n*n;
  if( ( R = (double*) malloc(N*sizeof(double)) ) == NULL ) {
    fprintf(stderr,"Error en la reserva de memoria para la matriz A\n");
    exit(-1);
  }
  srand(1234);
  /* Generamos aleatoriamente la parte triangular superior */
  for( i=0; i<n; i++ ) {
    for( j=i; j<n; j++ ) {
      R(i,j) = ((double) rand()) / RAND_MAX;
    }
  }

#ifdef DEBUG
  /* Imprimir matriz */
  printMatrix( "R = ", R, lda, n );
#endif

  /* Reservamos espacio para el vector independiente */
  if( ( b = (double*) malloc(n*sizeof(double)) ) == NULL ) {
    fprintf(stderr,"Error en la reserva de memoria para el vector b\n");
    exit(-1);
  }
  /* Generamos aleatoriamente el vector independiente */
  for( i=0; i<n; i++ ) {
    b(i) = ((double) rand()) / RAND_MAX;
  }

#ifdef DEBUG
  /* Imprimir vector */
  printf("b = [");
  for( i=0; i<n; i++ ) {
    printf("%10.4lf",b(i));
  }
  printf("    ] \n");
#endif

  /* Leemos por teclado el cardinal del conjunto discreto de elementos */
  sscanf(argv[2],"%d",&t);
  if( t%2 ) {
    fprintf(stderr,"El valor de t ha de ser par \n");
    exit(-1);
  }
  /* Reservamos espacio para el vector de elementos del conjunto discreto */
  if( ( I = (double*) malloc(t*sizeof(double)) ) == NULL ) {
    fprintf(stderr,"Error en la reserva de memoria para el conjunto A\n");
    exit(-1);
  }
  for( i=0; i<t; i++ ) {
    I(i) = (2*i+1-t)/2.0;
  }
#ifdef DEBUG
  /* Imprimir conjunto */
  printf("I = (");
  for( i=0; i<t; i++ ) {
    printf("%10.2lf",I(i));
  }
  printf("    )\n");
#endif

  /* Reservamos espacio para el vector solucion */
  if( ( x = (double*) malloc(n*sizeof(double)) ) == NULL ) {
    fprintf(stderr,"Error en la reserva de memoria para el vector x\n");
    exit(-1);
  }

  /* Llamada a la funcion que minimiza el problema de minimos cuadrados */
  minimo = 1.0/0.0; //(double) RAND_MAX;
  sol = (double*) malloc(n*sizeof(double));
  double nrm = 0.0;
  double elapsed, ucpu, scpu;
  ctimer(&elapsed,&ucpu,&scpu);
  algoritmo1( n, x, t, I, R, lda, b, nrm );
  ctimer(&elapsed,&ucpu,&scpu);
  printf("Tiempo = %lf\n",elapsed);
  
  printf("Minimo = %lf\n",minimo);
  printf("sqrt(minimo) = %lf\n",sqrt(minimo));
  printf("x = (");
  for( i=0; i<n; i++ ) {
    printf("%10.2lf",sol(i));
  }
  printf("    )\n");
#ifdef CONTAR_TAREAS
  printf("tareas = %d\n",tareas);
#endif

  free(x);
  free(I);
  free(R);
  free(b);
  free(sol);
}

void printMatrix( const char *s, double *R, int lda, int n ) {
  int i, j;
  for( i=0; i<n; i++ ) {
    for( j=0; j<i; j++ ) {
      printf("%10.4lf",0.0);
    }
    for( j=i; j<n; j++ ) {
      printf("%10.4lf",R(i,j));
    }
    printf("\n");
  }
}

/**
    Algoritmo1 del enunciado de la práctica en el que se trabaj con el cuadrado de la norma y se va calculando según    .
    se sugiere en la misma memoria.
 
    n	(int)		Tamaño del subvector de x sobre el que se está trabajando, x(0:n-1). Las componentes x(n:lda) se han calculado ya.
    x	(double*)	Puntero al primer elemento del vector x (completo).
    t	(int)		Tamaño del conjunto discreto I de símbolos.
    I	(double*)	Puntero al primer elemento del vector donde se encuentran los elementos del conjunto discreto I.
    R	(double*)	Puntero al primer elemento de la matriz triangular superior R de tamaño lda x n.
    lda	(int)		(Leading dimension) Número de filas de la matriz R. Coincide con el tamaño de los vectores x y b en la primera llamada a la función.
    b	(double*)	Puntero al primer elemento del vector b (completo).
    nrm	(double)	Norma calculada hasta el momento de la llamada a este algoritmo.

    El vector x es de entrada/salida, los demás argumentos son solo de entrada. 
*/
void algoritmo1( int n, double *x, int t, double *I, double *R, int lda, double *b, double nrm ) {
  int k, inc = 1;
  if( n==0 ) {
    if( nrm < minimo ) {
      minimo = nrm;
      dcopy_( &lda, x, &inc, sol, &inc );
    }
  } else {
    for( k=0; k<t; k++ ) {
      int m = n-1;
      x(m) = I(k);
      double r = R(m,m)*x(m) - b(m);
      double norma = nrm + r*r; 
      if( norma < minimo ) {
        double *v = malloc(m*sizeof(double));
        double *y = malloc(lda*sizeof(double));
        dcopy_( &m, b, &inc, v, &inc );
        dcopy_( &lda, x, &inc, y, &inc );
        double alpha = -x(m);
        daxpy_( &m, &alpha, &R(0,m), &inc, v, &inc );
        algoritmo1( m, y, t, I, R, lda, v, norma );
        free(v);
        free(y);
      }
    }
  }
}

