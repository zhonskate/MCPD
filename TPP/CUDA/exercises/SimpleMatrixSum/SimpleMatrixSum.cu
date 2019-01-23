
/*************************************
 * Simple CUDA kernel for matrix sum *
 *************************************/

#include <stdio.h>

#define CUDA_SAFE_CALL( call ) {                                         \
 cudaError_t err = call;                                                 \
 if( cudaSuccess != err ) {                                              \
   fprintf(stderr,"CUDA: error occurred in cuda routine. Exiting...\n"); \
   exit(err);                                                            \
 } }

#define	A(i,j)		A[ (j) + ((i)*(n)) ]
#define	B(i,j) 		B[ (j) + ((i)*(n)) ]
#define	C(i,j) 		C[ (j) + ((i)*(n)) ]
#define	C_gpu(i,j) 	C_gpu[ (j) + ((i)*(n)) ]
#define	C_cpu(i,j) 	C_cpu[ (j) + ((i)*(n)) ]
#define	d_A(i,j) 	d_A[ (j) + ((i)*(n)) ]
#define	d_B(i,j) 	d_B[ (j) + ((i)*(n)) ]
#define	d_C(i,j) 	d_C[ (j) + ((i)*(n)) ]

/* This kernel computes a matrix addition. Each thread executing this kernel performs a matrix element sum */
__global__ void compute_kernel( unsigned int m, unsigned int n, float *d_A, float *d_B, float *d_C ) {

  /* Obtain the global matrix index accessed by the thread executing this kernel */
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  int i = blockIdx.y * blockDim.y + threadIdx.y;

  /* Perform the addition. Pay attention because probably not all the threads should perform the addition */ 
  if(i < m && j < n){
    d_C( i,j ) = d_A( i,j ) + d_B( i,j );
  }

}

int cu_matrix_sum( unsigned int m, unsigned int n, unsigned int block_rows, unsigned int block_cols, float *h_A, float *h_B, float *h_C ) {

  // Allocate device memory
  unsigned int mem_size = m * n * sizeof(float);
  float *d_A, *d_B, *d_C;
  CUDA_SAFE_CALL( cudaMalloc((void **) &d_A, mem_size ) );
  CUDA_SAFE_CALL( cudaMalloc((void **) &d_B, mem_size ) );
  CUDA_SAFE_CALL( cudaMalloc((void **) &d_C, mem_size ) );

  // Copy host memory to device 
  CUDA_SAFE_CALL( cudaMemcpy( d_A, h_A, mem_size, cudaMemcpyHostToDevice ) );
  CUDA_SAFE_CALL( cudaMemcpy( d_B, h_B, mem_size, cudaMemcpyHostToDevice ) );

  int row_blocks = (int) ceil( (float) m / (float) block_rows );
  int col_blocks = (int) ceil( (float) n / (float) block_cols );

  // Execute the kernel
  dim3 dimGrid( row_blocks, col_blocks );
  dim3 dimBlock( block_rows, block_cols );
  compute_kernel<<< dimGrid, dimBlock >>>( m, n, d_A, d_B, d_C );

  // Copy device memory to host 
  CUDA_SAFE_CALL( cudaMemcpy( h_C, d_C, mem_size, cudaMemcpyDeviceToHost ) );

  // Deallocate device memory
  CUDA_SAFE_CALL( cudaFree(d_A) );
  CUDA_SAFE_CALL( cudaFree(d_B) );
  CUDA_SAFE_CALL( cudaFree(d_C) );

  return EXIT_SUCCESS;
}
 
int matrix_sum( unsigned int m, unsigned int n, float *A, float *B, float *C ) {

  unsigned int i, j;
  for( i=0; i<m; i++ ) {
    for( j=0; j<n; j++ ) {
      C( i, j ) = A( i, j ) + B( i, j );
    }
  }
  return EXIT_SUCCESS;

}

int main( int argc, char *argv[] ) {
  unsigned int m, n;
  unsigned int block_rows, block_cols;
  unsigned int i, j;

  /* Generating input data */
  if( argc<5 ) {
    printf("Usage: %s n_rows n_cols block_rows block_cols \n",argv[0]);
    exit(-1);
  }
  sscanf(argv[1],"%d",&m);
  sscanf(argv[2],"%d",&n);
  sscanf(argv[3],"%d",&block_rows);
  sscanf(argv[4],"%d",&block_cols);
  float *A = (float *) malloc( m*n*sizeof(float) );
  float *B = (float *) malloc( m*n*sizeof(float) );
  printf("%s: Generating two random matrices of size %dx%d...\n",argv[0],m,n);
  for( i=0; i<m; i++ ) {
    for( j=0; j<n; j++ ) {
      A( i, j ) = 2.0f * ( (float) rand() / RAND_MAX ) - 1.0f;
      B( i, j ) = 2.0f * ( (float) rand() / RAND_MAX ) - 1.0f;
    }
  }

  printf("%s: Adding matrices in CPU...\n",argv[0]);
  float *C_cpu = (float *) malloc( m*n*sizeof(float) );
  matrix_sum( m, n, A, B, C_cpu );

  printf("%s: Adding matrices in GPU...\n",argv[0]);
  float *C_gpu = (float *) malloc( m*n*sizeof(float) );
  cu_matrix_sum( m, n, block_rows, block_cols, A, B, C_gpu );

  /* Check for correctness */
  float error = 0.0f;
  for( i=0; i<m; i++ ) {
    for( j=0; j<n; j++ ) {
      error += fabs( C_gpu( i, j ) - C_cpu( i, j ) );
    }
  }
  printf("Error CPU/GPU = %.3e\n",error);
  
  free(A);
  free(B);
  free(C_cpu);
  free(C_gpu);
  
}

