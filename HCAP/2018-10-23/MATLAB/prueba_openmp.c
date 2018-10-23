#include "mex.h"
#include <omp.h>

 void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
 {
     int i;
     int nump=omp_get_max_threads();

     #pragma omp parallel for
     for (i = 0; i < nump; i++) 
        mexPrintf("Num threads %d, thread ID %d.\n", omp_get_num_threads(), omp_get_thread_num());
 }