

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
void matprod(double *A, double *B, double *C, mwSize n)
{
    mwSize i,j,k;
    /* multiply each element y by x */
    for (j=0; j<n; j++) 
         for(k=0;k<n; k++)
             for(i=0;i<n; i++)
       {
        C[i+n*j]= C[i+n*j]+ A[i+n*k]* B[k+n*j];
    }
}

/* The gateway function */
void mexFunction( int nlhs, mxArray *plhs[],
                  int nrhs, const mxArray *prhs[])
{
    double *inMatrixA;              /* matrix A*/
    double *inMatrixB;              /* matrix B*/
    size_t nrowsA;                  /* rows A*/
    size_t ncolsB;                  /* columns B*/
    double *outMatrix;              /* matrix C*/

    
    /* get the value of the scalar input  */
    nrowsA = mxGetM(prhs[0]);

    /* create a pointer to the real data in the input matrix  */
    inMatrixA = mxGetPr(prhs[0]);

    /* create a pointer to the real data in the input matrix  */
    inMatrixB = mxGetPr(prhs[1]);

    /* get dimensions of the input matrix */
    ncolsB = mxGetN(prhs[1]);

    /* create the output matrix */
    plhs[0] = mxCreateDoubleMatrix((mwSize)nrowsA,(mwSize)ncolsB,mxREAL);

    /* get a pointer to the real data in the output matrix */
    outMatrix = mxGetPr(plhs[0]);

    /* call the computational routine */
    matprod(inMatrixA,inMatrixB,outMatrix,(mwSize)ncolsB);
}