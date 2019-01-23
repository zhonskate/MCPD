
NVCC=nvcc
ARCH=-arch sm_20
CUDAFLAGS=$(ARCH) -lineinfo 
CUBLAS=-lcublas
BLAS=-Xcompiler -m64 -I$(MKLROOT)/include -Xlinker -L$(MKLROOT)/lib/intel64 -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lm

all: MatrixMatrixMultiplication 

MatrixMatrixMultiplication: MatrixMatrixMultiplication.cu 
	$(NVCC) $(CUDAFLAGS) $(BLAS) $(CUBLAS) MatrixMatrixMultiplication.cu -o MatrixMatrixMultiplication 

clean:
	rm -f *.o MatrixMatrixMultiplication.linkinfo MatrixMatrixMultiplication 

