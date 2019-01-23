
/*************************************
 * Matrix-Vector product CUDA kernel *
 * V1: Without Shared memory         *
 *************************************/

#include <stdio.h>

#define CUDA_SAFE_CALL( call ) {                                         \
 cudaError_t err = call;                                                 \
 if( cudaSuccess != err ) {                                              \
   fprintf(stderr,"CUDA: error occurred in cuda routine. Exiting...\n"); \
   exit(err);                                                            \
 } }

#define	A(i,j)		A[ (j) + ((i)*(n)) ]
#define	x(i) 		x[ (i) ]
#define	y(i) 		y[ (i) ]
#define	y_gpu(i) 	y_gpu[ (i) ]
#define	y_cpu(i) 	y_cpu[ (i) ]
#define	d_A(i,j) 	d_A[ (j) + ((i)*(n)) ]
#define	d_x(i) 		d_x[ (i) ]
#define	d_y(i) 		d_y[ (i) ]

#define	BLOCKSIZE	32

__global__ void compute_kernel( unsigned int m, unsigned int n, float *d_A, float *d_x, float *d_y ) {

    /* Figure out the global index to a matrix row */

    /* Each thread computes the ith component of array d_y by multiplying the ith row of d_A by array d_x */
    /* Pay attention again to if the component accessed by the thread falls within the bound */

}

int cu_matrix_vector( unsigned int m, unsigned int n, float *h_A, float *h_x, float *h_y ) {

  // Allocate device memory
  float *d_A, *d_x, *d_y;
  CUDA_SAFE_CALL( cudaMalloc((void **) &d_A, m*n*sizeof(float) ) );
  CUDA_SAFE_CALL( cudaMalloc((void **) &d_x,   n*sizeof(float) ) );
  CUDA_SAFE_CALL( cudaMalloc((void **) &d_y, m  *sizeof(float) ) );

  // Copy host memory to device 
  CUDA_SAFE_CALL( cudaMemcpy( d_A, h_A, m*n*sizeof(float), cudaMemcpyHostToDevice ) );
  CUDA_SAFE_CALL( cudaMemcpy( d_x, h_x,   n*sizeof(float), cudaMemcpyHostToDevice ) );

  int n_blocks = (int) ceil( (float) m / (float) BLOCKSIZE );

  // Execute the kernel
  dim3 dimGrid( n_blocks );
  dim3 dimBlock( BLOCKSIZE );
  compute_kernel<<< dimGrid, dimBlock >>>( m, n, d_A, d_x, d_y );

  // Copy device memory to host 
  CUDA_SAFE_CALL( cudaMemcpy( h_y, d_y, m  *sizeof(float), cudaMemcpyDeviceToHost ) );

  // Deallocate device memory
  CUDA_SAFE_CALL( cudaFree(d_A) );
  CUDA_SAFE_CALL( cudaFree(d_x) );
  CUDA_SAFE_CALL( cudaFree(d_y) );

  return EXIT_SUCCESS;
}
 
int matrix_vector( unsigned int m, unsigned int n, float *A, float *x, float *y ) {

  unsigned int i, j;
  for( i=0; i<m; i++ ) {
    y( i ) = 0.0f;
    for( j=0; j<n; j++ ) {
      y( i ) += A( i, j ) * x( j );
    }
  }
  return EXIT_SUCCESS;

}

int main( int argc, char *argv[] ) {
  unsigned int m, n;
  unsigned int i, j;

  /* Generating input data */
  if( argc<3 ) {
    printf("Usage: %s n_rows n_cols \n",argv[0]);
    exit(-1);
  }
  sscanf(argv[1],"%d",&m);
  sscanf(argv[2],"%d",&n);
  float *A = (float *) malloc( m*n*sizeof(float) );
  float *x = (float *) malloc(   n*sizeof(float) );
  printf("%s: Generating a random matrix of size %dx%d and a vector of size %d...\n",argv[0],m,n,n);
  for( i=0; i<m; i++ ) {
    for( j=0; j<n; j++ ) {
      A( i, j ) = 2.0f * ( (float) rand() / RAND_MAX ) - 1.0f;
    }
  }
  for( j=0; j<n; j++ ) {
    x( j ) = 2.0f * ( (float) rand() / RAND_MAX ) - 1.0f;
  }


  // Allocate CUDA events that we'll use for timing
  cudaEvent_t start, stop;
  CUDA_SAFE_CALL( cudaEventCreate(&start) );
  CUDA_SAFE_CALL( cudaEventCreate(&stop) );

  printf("%s: y=A*x in CPU...\n",argv[0]);
  float *y_cpu = (float *) malloc( m*sizeof(float) );
  CUDA_SAFE_CALL( cudaEventRecord(start, NULL) ); // Record the start event
  matrix_vector( m, n, A, x, y_cpu );
  CUDA_SAFE_CALL( cudaEventRecord(stop, NULL) );  // Record the stop event
  CUDA_SAFE_CALL( cudaEventSynchronize(stop) );   // Wait for the stop event to complete
  float msecCPU = 0.0f;
  CUDA_SAFE_CALL( cudaEventElapsedTime(&msecCPU, start, stop) );

  printf("%s: y=A*x in GPU...\n",argv[0]);
  float *y_gpu = (float *) malloc( m*sizeof(float) );
  CUDA_SAFE_CALL( cudaEventRecord(start, NULL) ); // Record the start event
  cu_matrix_vector( m, n, A, x, y_gpu );
  CUDA_SAFE_CALL( cudaEventRecord(stop, NULL) );  // Record the stop event
  CUDA_SAFE_CALL( cudaEventSynchronize(stop) );   // Wait for the stop event to complete
  float msecGPU = 0.0f;
  CUDA_SAFE_CALL( cudaEventElapsedTime(&msecGPU, start, stop) );

  /* Check for correctness */
  float max = fabs( y_cpu( 0 ) );
  for( i=1; i<m; i++ ) {
    max = fabs( y_cpu( i ) > max ? y_cpu( i ) : max );
  }
  float error = 0.0f;
  for( i=0; i<m; i++ ) {
    error += fabs( y_gpu( i ) - y_cpu( i ) );
  }
  printf("Error CPU/GPU = %.3e\n",error/max);
  double flops = 2.0 * (double) m * (double) n;
  double gigaFlopsCPU = (flops * 1.0e-9f) / (msecCPU / 1000.0f);
  double gigaFlopsGPU = (flops * 1.0e-9f) / (msecGPU / 1000.0f);
  printf("CPU time = %.2f msec.\n",msecCPU);
  printf("GPU time = %.2f msec.\n",msecGPU);
  printf("Gflops CPU = %.2f \n",gigaFlopsCPU);
  printf("Gflops GPU = %.2f \n",gigaFlopsGPU);
  
  free(A);
  free(x);
  free(y_cpu);
  free(y_gpu);
  
}

