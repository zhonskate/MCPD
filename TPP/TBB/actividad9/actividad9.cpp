
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

/*double SerialSumFoo( double a[], size_t n ) { 
  double sum = 0;
  for( size_t i=0; i!=n; ++i ) 
    sum += Foo(a[i]);
  return sum;
}*/

Class SumFoo {

  double* my_a;

  public:
    double sum;
    void operator()(const blocked_range<size_t>& r){
      double *a = my_a;
      for( size_t i=r.begin(); i!=r.end(); ++i ){
        sum = sum + Foo(a[i]);
      }
    }
    SumFoo( SumFoo m, split) : my_a(m.my_a),sum(0){}
    void join( const SumFoo& n){sum = sum + y.sum;}
    SumFoo(double a[]) : my_a(a), sum(0){}
};

double ParaSumFoo(double a[], size_t n ) {
  SumFoo sf(a);
  parallel_reduce( blocked_range<size_t>(0,n,1000),sf);	
  return sf.sum;	
}

int main( )  {

  long int n = 1000000;
  double *A = (double *) malloc( n*sizeof(double) );
  for( size_t i=0; i<n; ++i ) A[i] = i;
  double suma = ParaSumFoo( A, n );
  cout << "Suma = " << suma << endl;

}


