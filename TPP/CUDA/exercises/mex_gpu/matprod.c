
#include <mex.h>

void matprod( int m, int n, int p, double A[], double B[], double C[] ) {
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

  matprod(m,n,p,A,B,C);

}

