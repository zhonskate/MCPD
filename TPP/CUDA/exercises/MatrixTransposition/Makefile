
NVCC=nvcc
ARCH=-arch sm_20
CUDAFLAGS=$(ARCH) -lineinfo

all: MatrixTransposition_v1 MatrixTransposition_v2

MatrixTransposition_v1: MatrixTransposition_v1.cu 
	$(NVCC) MatrixTransposition_v1.cu -o MatrixTransposition_v1 $(CUDAFLAGS) 

MatrixTransposition_v2: MatrixTransposition_v2.cu 
	$(NVCC) MatrixTransposition_v2.cu -o MatrixTransposition_v2 $(CUDAFLAGS) 

clean:
	rm -f *.o MatrixTransposition_v1.linkinfo MatrixTransposition_v1 MatrixTransposition_v2.linkinfo MatrixTransposition_v2

