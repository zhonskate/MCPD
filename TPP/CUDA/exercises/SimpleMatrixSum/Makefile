
NVCC=nvcc

ARCH=-arch sm_20
CUDAFLAGS=-lineinfo $(ARCH)

all: SimpleMatrixSum

SimpleMatrixSum: SimpleMatrixSum.cu 
	$(NVCC) SimpleMatrixSum.cu -o SimpleMatrixSum $(CUDAFLAGS) 

clean:
	rm -f *.o SimpleMatrixSum.linkinfo SimpleMatrixSum

