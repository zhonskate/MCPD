
CUDA_LIB_PATH=/usr/local/cuda/lib64
CUDA_LIB=cudart
CUBLAS=cublas
MATLAB_INCLUDE_PATH=/opt/matlab/r2012b/extern/include

cumatprod.mexa64: cumatprod.o
	mex cumatprod.o -cxx -L$(CUDA_LIB_PATH) -l$(CUDA_LIB) -l$(CUBLAS)

cumatprod.o: cumatprod.cu
	nvcc -c cumatprod.cu -I$(MATLAB_INCLUDE_PATH) --compiler-options -fPIC

clean:
	rm -f cumatprod.mexa64 cumatprod.o
