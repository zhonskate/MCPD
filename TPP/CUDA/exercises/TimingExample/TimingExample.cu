
/************************************************
 * Simple CUDA example to transfer data CPU-GPU *
 ************************************************/

#include <stdio.h>
#include <stdlib.h>

#define CUDA_SAFE_CALL( call ) {                                         \
 cudaError_t err = call;                                                 \
 if( cudaSuccess != err ) {                                              \
   fprintf(stderr,"CUDA: error occurred in cuda routine. Exiting...\n"); \
   exit(err);                                                            \
 } }

#define	A(i,j)		A[ (j) + ((i)*(n)) ]
#define	B(i,j) 		B[ (j) + ((i)*(n)) ]

int main( int argc, char *argv[] ) {
  unsigned int m, n;
  unsigned int i, j;

  /* Generating input data */
  if( argc<3 ) {
    printf("Usage: %s rows cols \n",argv[0]);
    exit(-1);
  }
  sscanf(argv[1],"%d",&m);
  sscanf(argv[2],"%d",&n);

  /* STEP 1: Allocate memory for three m-by-n matrices called A and B in the host */
  float *A, *B, *C;
  A = (float*) malloc( m*n*sizeof(float) );
  C = (float*) malloc( m*n*sizeof(float) );
  B = (float*) malloc( m*n*sizeof(float) );

  for(int i=0;i<m;i++){
    for(int j=0;j<n;j++){
      A(i,j) = ( 2.0f * (float) rand() / 1.0f ) - 1.0f;
    }
  }

  /* STEP 2: Fill matrices A and B with real values between -1.0 and 1.0 */
  float *d_A, *d_B, *d_C;

  cudaEvent_t start, stop;
  float elapsedTime;

  CUDA_SAFE_CALL( cudaEventCreate(&start) );
  CUDA_SAFE_CALL( cudaEventRecord(start,0) );

  /* STEP 3: Allocate memory for three m-by-n matrices into the device memory */
  CUDA_SAFE_CALL( cudaMalloc( (void**) &d_A, m*n*sizeof(float) ) );
  CUDA_SAFE_CALL( cudaMalloc( (void**) &d_B, m*n*sizeof(float) ) );
  CUDA_SAFE_CALL( cudaMalloc( (void**) &d_C, m*n*sizeof(float) ) );
  
  /* STEP 4: Copy host memory (only matrices A and B) to the device memory (matrices d_A and d_B) */
  CUDA_SAFE_CALL( cudaMemcpy(d_A, A, m*n*sizeof(float), cudaMemcpyHostToDevice) );
  CUDA_SAFE_CALL( cudaMemcpy(d_B, B, m*n*sizeof(float), cudaMemcpyHostToDevice) );


  /* STEP 5: Copy back from device memory into the host memory only data cohrresponding to matrix C (d_C) */
  CUDA_SAFE_CALL( cudaMemcpy(C, d_C, m*n*sizeof(float), cudaMemcpyDeviceToHost) );

  /* STEP 6: Deallocate device memory */
  CUDA_SAFE_CALL( cudaFree(d_A) );
  CUDA_SAFE_CALL( cudaFree(d_B) );
  CUDA_SAFE_CALL( cudaFree(d_C) );

    
  CUDA_SAFE_CALL( cudaEventCreate(&stop) );
  CUDA_SAFE_CALL( cudaEventRecord(stop,0) );
  CUDA_SAFE_CALL( cudaEventSynchronize(stop) );
  CUDA_SAFE_CALL( cudaEventElapsedTime(&elapsedTime, start,stop); );

  printf("Time: %f\n", elapsedTime);

  /* STEP 7: Deallocate host memory */
  free(A);
  free(B);
  free(C);
}

