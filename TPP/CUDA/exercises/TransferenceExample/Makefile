
NVCC=nvcc

ARCH=-arch sm_20
CUDAFLAGS=-lineinfo $(ARCH)

all: TransferenceExample

TransferenceExample: TransferenceExample.cu 
	$(NVCC) TransferenceExample.cu -o TransferenceExample $(CUDAFLAGS) 

clean:
	rm -f *.o TransferenceExample.linkinfo TransferenceExample

