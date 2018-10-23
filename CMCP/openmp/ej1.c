#include <stdio.h>
#include <math.h>
#include <float.h>
#include <stdlib.h>
#include <omp.h>
#include "ctimer.h"
main(int argc, char**argv)
{
    int tid;
    int nthreads;
    #pragma omp parallel private(tid,nthreads)
    {
        nthreads = omp_get_num_threads();
        tid = omp_get_thread_num();
        printf("soy el thread %u de %u \n", tid, nthreads);
    }
}