/*=========================================================
 * matrixDivide64.c - Example for illustrating how to use 
 * LAPACK within a C MEX-file. MOdificado para SO de 64 bits
 *
 * X = matrixDivide(A,B) computes the solution to a 
 * system of linear equations A * X = B
 * using LAPACK routine DGESV, where 
 * A is an N-by-N matrix  
 * X and B are N-by-1 matrices.
 *
 * This is a MEX-file for MATLAB.
 * Copyright 2009-2010 The MathWorks, Inc.
 *=======================================================*/
/* $Revision: 1.1.6.3 $ $Date: 2011/01/28 18:11:56 $ */

#if !defined(_WIN32)
#define dgesv dgesv_
#endif

#include "mex.h"
#include "lapack.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    double *A;    /* pointers to input matrices */
    double *A2;  /* in/out arguments to DGESV*/
    size_t m,n;     /* matrix dimensions */ 
    mwSignedIndex *iPivot;   /* inputs to DGESV */
    mxArray *Awork, *mxPivot;
    mwSignedIndex info, dims[1];
  
 	/* Check for proper number of arguments. */

    A = mxGetPr(prhs[0]); /* pointer to first input matrix */
    /* dimensions of input matrices */
    m = mxGetM(prhs[0]);  
    n = mxGetN(prhs[0]);

    /* Validate input arguments */
    if (n != m) {
        mexErrMsgIdAndTxt("MATLAB:matrixDivide64:square",
            "LAPACK function requires input matrix 1 must be square.");
    }

    /* DGESV works in-place, so we copy the inputs first. */
    plhs[0] = mxCreateDoubleMatrix(m, n, mxREAL);
    A2 = mxGetPr(plhs[0]);
    memcpy(A2, A, m*n*mxGetElementSize(prhs[0]));
  
    /* Create inputs for DGESV */
    dims[0] = m;
    plhs[1] = mxCreateNumericArray(1, dims, mxINT64_CLASS, mxREAL);
    iPivot = (mwSignedIndex*)mxGetData(plhs[1]);
  
    /* Call LAPACK */
    dgetrf(&m,&m,A2,&m,iPivot,&info);
    /* plhs[0] now holds X */
}
