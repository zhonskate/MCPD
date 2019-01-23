
NVCC=nvcc
ARCH=-arch sm_20
CUDAFLAGS=$(ARCH) -lineinfo

all: MatrixVectorProduct_v1 MatrixVectorProduct_v2

MatrixVectorProduct_v1: MatrixVectorProduct_v1.cu 
	$(NVCC) MatrixVectorProduct_v1.cu -o MatrixVectorProduct_v1 $(CUDAFLAGS) 

MatrixVectorProduct_v2: MatrixVectorProduct_v2.cu 
	$(NVCC) MatrixVectorProduct_v2.cu -o MatrixVectorProduct_v2 $(CUDAFLAGS) 

clean:
	rm -f *.o MatrixVectorProduct_v1.linkinfo MatrixVectorProduct_v1 MatrixVectorProduct_v2.linkinfo MatrixVectorProduct_v2

