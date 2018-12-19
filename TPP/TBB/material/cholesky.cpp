
/* 
 * Cholesky por bloques. 
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "ctimer.h"

#define	L(i,j)	L[j*n+i]
#define	A(i,j)	A[j*n+i]
#define	C(i,j)	C[j*n+i]

#ifdef __cplusplus
extern "C" {
#endif
void dpotrf_( const char* uplo, const int* n, double* a, const int* lda, int* info );
void dgemm_(const char *transa, const char *transb, const int *m, const int *n, const int *k,
            const double *alpha, const double *a, const int *lda, const double *b, const int *ldb,
            const double *beta, double *c, const int *ldc);
void dsyrk_(const char *uplo, const char *trans, const int *n, const int *k,
            const double *alpha, const double *a, const int *lda, const double *beta,
            double *c, const int *ldc);
void dtrsm_(const char *side, const char *uplo, const char *transa, const char *diag,
            const int *m, const int *n, const double *alpha, const double *a, const int *lda,
            double *b, const int *ldb);
#ifdef __cplusplus
}
#endif

int cholesky( int n, int b, double *C );

int main( int argc, char *argv[] ) {
  int n, b, i, j, info;
  double *L, *A;

  if( argc<3 ) {
    fprintf(stderr,"usage: %s n block_size\n",argv[0]);
    exit(-1);
  }
  sscanf(argv[1],"%d",&n);
  if( ( L = (double*) malloc(n*n*sizeof(double)) ) == NULL ) {
    fprintf(stderr,"Error en la reserva de memoria para la matriz L\n");
    exit(-1);
  }
  for( j=0; j<n; j++ ) {
    for( i=0; i<j; i++ ) {
      L(i,j) = 0.0;
    }
    for( i=j; i<n; i++ ) {
      L(i,j) = ((double) rand()) / RAND_MAX;
    }
    L(j,j) += n;
  }
  /* Imprimir matriz */
  /*
  for( i=0; i<n; i++ ) {
    for( j=0; j<n; j++ ) {
      printf("%10.3lf",L(i,j));
    }
    printf("\n");
  }
  */
  if( ( A = (double*) malloc(n*n*sizeof(double)) ) == NULL ) {
    fprintf(stderr,"Error en la reserva de memoria para la matriz A\n");
    exit(-1);
  }

  /*********************************************************/
  /* Multiplicación A=L*L', donde L es triangular inferior */
  /* Devuelve la parte triangular inferior en A */
  double zero = 0.0;
  double one = 1.0;
  dsyrk_( "L", "N", &n, &n, &one, &L(0,0), &n, &zero, &A(0,0), &n );
  /*********************************************************/

  sscanf(argv[2],"%d",&b);

  /* Imprimir matriz */
  /*
  for( i=0; i<n; i++ ) {
    for( j=0; j<n; j++ ) {
      printf("%10.3lf",A(i,j));
    }
    printf("\n");
  }
  */

  double t1, t2, ucpu, scpu;
  ctimer( &t1, &ucpu, &scpu );
  info = cholesky( n, b, A );
  //dpotrf_( "L", &n, A, &n, &info ); 
  ctimer( &t2, &ucpu, &scpu );
  if( info != 0 ) {
    fprintf(stderr,"Error = %d en la descomposición de Cholesky de la matriz A\n",info);
    exit(-1);
  }

  /* Imprimir matriz */
  /*
  for( i=0; i<n; i++ ) {
    for( j=0; j<n; j++ ) {
      printf("%10.3lf",A(i,j));
    }
    printf("\n");
  }
  */

  /* ¿ A = L ? */
  double error = 0.0;
  for( j=0; j<n; j++ ) {
    for( i=j; i<n; i++ ) {
      double b = (A(i,j)-L(i,j));
      error += b*b;
    }
  }
  error = sqrt(error);
  //printf("Error = %10.4e\n",error);
  printf("%10d %10d %20.2f sec. %15.4e\n",n,b,t2-t1,error);

  free(A);
  free(L);
}

inline int min(int a, int b) { return (a < b) ? a : b; }

int cholesky( int n, int b, double *C ) {
  int i, j, k, m;
  int info;
  const double one = 1.0;
  const double minusone = -1.0;
  for ( k = 0; k < n ; k+=b ) {
    m = min( n-k, b );
    dpotrf_( "L", &m, &C(k,k), &n, &info );
    if( info != 0 ) {
      fprintf(stderr,"Error = %d en la descomposición de Cholesky de la matriz C\n",info);
      return info;
    }
    for ( i = k + b; i < n; i += b ) {
      m = min( n-i, b );
      dtrsm_( "R", "L", "T", "N", &m, &b, &one, &C(k,k), &n, &C(i,k), &n );
    }
    for ( i = k + b; i < n; i += b ) {
      m = min( n-i, b );
      for ( j = k + b; j < i ; j += b ) {
        dgemm_( "N", "T", &m, &b, &b, &minusone, &C(i,k), &n, &C(j,k), &n, &one, &C(i,j), &n );
      }
      dsyrk_( "L", "N", &m, &b, &minusone, &C(i,k), &n, &one, &C(i,i), &n );
    }
  }
  return 0;
}

