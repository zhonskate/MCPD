
#include <stdlib.h>
#include <cfloat>
#include <iostream>
#include "tbb/parallel_reduce.h"
#include "tbb/blocked_range.h"
using namespace std;
using namespace tbb;

double Foo( double f ) { 
  double a = f*f; 
  return a;
}

class MinIndexFoo {
  const double *const my_a;
  public:
    double value_of_min;
    long index_of_min;
  void operator()( const blocked_range<size_t>& r){
    const double *a = my_a;
    for( size_t i = r.begin(); i != r.end(); i++){
      double value = Foo(a[i]);
      if( value < value_of_min){
        value_of_min = value;
        index_of_min = i;
	    }
	  }	
  }
  MinIndexFoo(const double a[]): my_a(a), value_of_min(FLT_MAX), index_of_min(-1){}

  MinIndexFoo( MinIndexFoo& x, split) :my_a(x.my_a), value_of_min(FLT_MAX), index_of_min(-1){}

  void join(const MinIndexFoo& z){
	  if(z.value_of_min < value_of_min){
	    value_of_min = z.value_of_min;
	    index_of_min = z.index_of_min;
	  }
  }				
};

long ParaMinIndexFoo(double a[], size_t n){ 
  MinIndexFoo minif(a);
  parallel_reduce(blocked_range<size_t>(0,n,1000),minif);
  return minif.index_of_min;
}

/*long SerialMinIndexFoo( const double a[], size_t n ) {
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
}*/

int main( )  {

  long int n = 1000000;
  double *A = (double *) malloc( n*sizeof(double) );
  for( size_t i=0; i<n; ++i ) A[i] = (double) rand() / RAND_MAX;
  long indice = ParaMinIndexFoo( A, n );
  cout << "Minimo número = " << A[indice] << endl;

}

