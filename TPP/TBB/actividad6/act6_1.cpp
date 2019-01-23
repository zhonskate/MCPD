#include <stdlib.h>
#include <iostream>
#include "tbb/blocked_range.h"
#include "tbb/parallel_for.h"
using namespace std;
using namespace tbb;
 
void Foo( double& f ) { 
  unsigned int u = (unsigned int) f;
  f = (double) rand_r(&u) / RAND_MAX;
  f = f*f; 
}

class ApplyFoo {
  double *const my_a;
  public:
    void operator()( const blocked_range<size_t>& r ) const {
      double *a = my_a;
      for( size_t i=r.begin(); i!=r.end(); ++i )
        Foo(a[i]);
      }
    ApplyFoo( double a[] ) : my_a(a) {}
};

void SerialApplyFoo( double a[], size_t n ) { 
  for( size_t i=0; i<n; ++i )
    Foo(a[i]);
}

void ParallelApplyFoo( double a[], size_t n ) {
  //parallel_for( blocked_range<size_t>( 0, n), ApplyFoo(a) );
  parallel_for( blocked_range<size_t>( 0, n), [=](const blocked_range<size_t>& r ) {
      for( size_t i=r.begin(); i!=r.end(); ++i )
        Foo(a[i]);
  } );
}

int main( )  {

  //long int n = 1000000;
  long int n = 10;
  double *A = (double *) malloc( n*sizeof(double) );
  for( size_t i=0; i<n; ++i ) A[i] = i;
  //SerialApplyFoo( A, n );
  ParallelApplyFoo( A, n );
  for( size_t i=0; i<n; ++i )
    cout << A[i] << endl;
}