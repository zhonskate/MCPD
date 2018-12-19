
#include <stdlib.h>
#include <cfloat>
#include <iostream>
using namespace std;

double Foo( double f ) { 
  double a = f*f; 
  return a;
}

long SerialMinIndexFoo( const double a[], size_t n ) {
  double value_of_min = RAND_MAX; 
  long index_of_min = -1; 
  for( size_t i=0; i<n; ++i ) {
    double value = Foo(a[i]); 
    if( value < value_of_min ) {
      value_of_min = value; 
      index_of_min = i;
    }
  } 
  return index_of_min;
}

int main( )  {

  long int n = 1000000;
  double *A = (double *) malloc( n*sizeof(double) );
  for( size_t i=0; i<n; ++i ) A[i] = (double) rand() / RAND_MAX;
  long indice = SerialMinIndexFoo( A, n );
  cout << "Minimo número = " << A[indice] << endl;

}

