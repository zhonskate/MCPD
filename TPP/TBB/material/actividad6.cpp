
#include <stdlib.h>
#include <iostream>
using namespace std;
 
void Foo( double& f ) { 
  unsigned int u = (unsigned int) f;
  f = (double) rand_r(&u) / RAND_MAX;
  f = f*f; 
}

void SerialApplyFoo( double a[], size_t n ) { 
  for( size_t i=0; i<n; ++i )
    Foo(a[i]);
}

int main( )  {

  long int n = 1000000;
  double *A = (double *) malloc( n*sizeof(double) );
  for( size_t i=0; i<n; ++i ) A[i] = i;
  SerialApplyFoo( A, n );
  //for( size_t i=0; i<n; ++i )
  //  cout << A[i] << endl;
}

