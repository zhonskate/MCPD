
#include <stdio.h>
#include <stdlib.h>
#include <tbb/parallel_invoke.h>
using namespace tbb;

void fib(size_t n, size_t& m );

class FibClass {
  size_t n, &m;
  public: 
    FibClass( size_t n, size_t& m ) : n(n), m(m) { }
    void operator()() const {
      fib( n, m );
    }
};

void fib(size_t n, size_t& m ) { 
   size_t i, j; 
   if (n<2) 
     m = n; 
   else { 
         FibClass f1( n-1, i );
         FibClass f2( n-2, j );
         parallel_invoke( f1, f2 ); 
         m = i+j;
   } 
}

size_t fib(size_t n) {
  size_t i, j;
  if (n<2)
    return n;
  else {
    parallel_invoke( [&]{i=fib(n-1);},
    [&]{j=fib(n-2);} );
    return i+j;
  }
}

int main( int argc, char *argv[] ) {

  int n;
  size_t m;
  if( argc<2 ) {
    printf("Usage: %s n\n",argv[0]);
    exit(0);
  }
  sscanf(argv[1],"%d",&n);

  //fib( (size_t)n, m );
  m = fib( (size_t)n);
  printf("m = %ld\n",m);
}

