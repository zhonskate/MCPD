
NVCC=nvcc
ARCH=-arch sm_20
CUDAFLAGS=$(ARCH) -lineinfo

all: MatMulStreams 

MatMulStreams: MatMulStreams.cu 
	$(NVCC) MatMulStreams.cu -o MatMulStreams $(CUDAFLAGS) -lcublas -lblas

clean:
	rm -f *.o MatMulStreams.linkinfo MatMulStreams 

