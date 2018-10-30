

/*==========================================================
 * arrayProduct.c - example in MATLAB External Interfaces
 *
 * Multiplies an input scalar (multiplier) 
 * times a 1xN matrix (inMatrix)
 * and outputs a 1xN matrix (outMatrix)
 *
 * The calling syntax is:
 *
 *		outMatrix = arrayProduct(multiplier, inMatrix)
 *
 * This is a MEX-file for MATLAB.
 * Copyright 2007-2010 The MathWorks, Inc.
 *
 *========================================================*/
/* $Revision: 1.1.10.3 $ */

#include "mex.h"

/* The computational routine */
void luc(double *A, mwSize n)
{
    mwSize i,j,k;
    /* multiply each element y by x */
    for (k=0; k<n-1; k++){
        if(A[k+k*n]==0){
            return;
        }
        for(i=k+1; i<n; i++){
            A[i+k*n]=A[i+k*n]/A[k+k*n];
        }
        for(j=k+1; j<n; j++){
            for(i=k+1; i<n; i++){
                A[i+j*n] = A[i+j*n]-A[k+j*n]*A[i+k*n];
            }
        }
    }
}

/* The gateway function */
void mexFunction( int nlhs, mxArray *plhs[],
                  int nrhs, const mxArray *prhs[])
{

    double *A,*C;               /* 1xN input matrix */
    size_t ncols;                   /* size of matrix */
    double *outMatrix;              /* output matrix */

    /* check for proper number of arguments */
    if(nrhs!=1) {
        mexErrMsgIdAndTxt("MyToolbox:arrayProduct:nrhs","Two inputs required.");
    }
    if(nlhs!=1) {
        mexErrMsgIdAndTxt("MyToolbox:arrayProduct:nlhs","One output required.");
    }
    /* make sure the first input argument is scalar */
 
    
    
    /* check that number of rows in second input argument is 1 */
  
    
  

    /* create a pointer to the real data in the input matrix  */
    A = mxGetPr(prhs[0]); 

    /* get dimensions of the input matrix */
    ncols = mxGetN(prhs[0]);

    /* create the output matrix */
    plhs[0] = mxCreateDoubleMatrix((mwSize)ncols,(mwSize)ncols,mxREAL);

    /* get a pointer to the real data in the output matrix */
    C = mxGetPr(plhs[0]);
    
    for(int i = 0;i<ncols;i++){
        for(int j = 0;j<ncols;j++){
            C[i+j*ncols]=A[i+j*ncols];
        }
    }

    /* call the computational routine */
    luc(C,(mwSize)ncols);
}

