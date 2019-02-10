
#include <mex.h>
#include <stdio.h>
#include <mkl_blas.h>
#include "cublas_v2.h" 
#include <cuda_runtime.h>

#define CUDA_SAFE_CALL( call ) {                                         \
  cudaError_t err = call;                                                 \
  if( cudaSuccess != err ) {                                              \
    fprintf(stderr,"CUDA: error occurred in cuda routine. Exiting...\n"); \
    exit(err);                                                            \
  } }
 #define CUBLAS_SAFE_CALL( call ) {                                         \
  cublasStatus_t err = call;                                                \
  if( CUBLAS_STATUS_SUCCESS != err ) {                                      \
    fprintf(stderr,"CUBLAS: error occurred in cuda routine. Exiting...\n"); \
    cublasDestroy(handle);                                                  \
    exit(err);                                                              \
  } }

void cumatprod( int m, int n, int p, double A[], double B[], double C[] ) {

  cublasHandle_t handle;
  CUBLAS_SAFE_CALL( cublasCreate(&handle) );

  const double ONE = 1.0;
  const double ZERO = 0.0;

  double *d_A, *d_B, *d_C;
  CUDA_SAFE_CALL( cudaMalloc((void **) &d_A, n*n*sizeof(double) ) );
  CUDA_SAFE_CALL( cudaMalloc((void **) &d_B, n*n*sizeof(double) ) );
  CUDA_SAFE_CALL( cudaMalloc((void **) &d_C, n*n*sizeof(double) ) );

  CUDA_SAFE_CALL(  cudaMemcpy(d_A, A, n*n*sizeof(float), cudaMemcpyHostToDevice) );
  CUDA_SAFE_CALL(  cudaMemcpy(d_B, B, n*n*sizeof(float), cudaMemcpyHostToDevice) );

  CUBLAS_SAFE_CALL( cublasDgemm( handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &ONE, d_A, n, d_B, n, &ZERO, d_C, n) );

  CUDA_SAFE_CALL( cudaMemcpy(C, d_C, n*n*sizeof(float), cudaMemcpyDeviceToHost) );

  CUBLAS_SAFE_CALL( cublasDestroy(handle) );
  free(A);
  free(B);
  free(C);

  int i, j, k;
  for( i=0; i<m; i++ ) {
    for( j=0; j<n; j++ ) {
      for( k=0; k<p; k++ ) {
        C[i+m*j] += A[i+m*k] * B[k+p*j];
      }
    }
  }
}

void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[] ) {
  double *A, *B, *C;
  size_t m, n, p;

  /* Check for proper number of arguments. */
  if(nrhs!=2) {
    mexErrMsgIdAndTxt("MATLAB:matprod:invalidNumInputs","One input required.");
  } else {
    if(nlhs>1) {
      mexErrMsgIdAndTxt("MATLAB:matprod:maxlhs","Too many output arguments.");
    }
  }

  /* The input must be a noncomplex scalar double.*/
  m = mxGetM(prhs[0]);
  n = mxGetN(prhs[1]);
  p = mxGetN(prhs[0]);

  if( p!=mxGetM(prhs[1]) ) {
    mexErrMsgIdAndTxt( "MATLAB:matprod:invalidDimensions", "Second dimension of first input must be equal to the first dimension of second input.");
  }

  /* Create matrix for the return argument. */
  plhs[0] = mxCreateDoubleMatrix((mwSize)m, (mwSize)n, mxREAL);
  /* Assign pointers to each input and output. */
  A = mxGetPr(prhs[0]);
  B = mxGetPr(prhs[1]);
  C = mxGetPr(plhs[0]);

  cumatprod(m,n,p,A,B,C);

}

