
#include <stdlib.h>
#include <iostream>
#include "tbb/blocked_range.h"
#include "tbb/parallel_for.h"

using namespace tbb;

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

class MinIndexFoo {
const float *const my_a;
public:
float value_of_min;
long index_of_min;
void operator()( const blocked_range<size_t>& r ) {
const float *a = my_a;
for( size_t i=r.begin(); i!=r.end(); ++i ) {
float value = Foo(a[i]);
if( value<value_of_min ) {
value_of_min = value;
index_of_min = i;
}
}
}

int main( )  {

  long int n = 1000000;
  double *A = (double *) malloc( n*sizeof(double) );
  for( size_t i=0; i<n; ++i ) A[i] = i;
  double suma = SerialSumFoo( A, n );
  cout << "Suma = " << suma << endl;

}


