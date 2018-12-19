
#include <stdlib.h>
#include <iostream>
using namespace std;

double Foo( double f ) { 
  unsigned int u = (unsigned int) f;
  double a = (double) rand_r(&u) / RAND_MAX;
  a = a*a; 
  return a;
}

double SerialSumFoo( double a[], size_t n ) { 
  double sum = 0;
  for( size_t i=0; i!=n; ++i ) 
    sum += Foo(a[i]);
  return sum;
}

int main( )  {

  long int n = 1000000;
  double *A = (double *) malloc( n*sizeof(double) );
  for( size_t i=0; i<n; ++i ) A[i] = i;
  double suma = SerialSumFoo( A, n );
  cout << "Suma = " << suma << endl;

}


