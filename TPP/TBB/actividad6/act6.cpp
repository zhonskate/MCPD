#include <stdlib.h>
#include <iostream>
#include <tbb/task_scheduler_init.h>

using namespace tbb;

void Foo (double& f) {
    unsigned int u = (unsigned int) f;
    f = (double) rand_r(&u)/ RAND_MAX;
    f = f*f;
}

class ApplyFoo {
    double *const my_a;
    public:
        void operator()(const blocked_range<size_t>& r) const {
            double *a = my_a;
            for(size_t i = r.begin(); i!=r.end(); ++i)
                Foo(a[i]);
        }
        ApplyFoo(double a[]) : my_a(a) {}
};

void serialApplyFoo (double a[], size_t n){
    for(size_t i=0; i<n; i++){
        Foo(a[i]);
    }
}

void parallelApplyFoo (double a[], size_t n){
    parallel_for( blocked_range<size_t>(0, n), ApplyFoo(a));

}

int main () {

    long int n = 10;
    double *A = (double *) malloc(n*sizeof(double));
    for(size_t i=0; i<n; ++i) A[i] = 1.0;
    serialApplyFoo(A,n);
    for (size_t i=0; i<n; ++i)
        cout << A[i] << endl;
}