
NVCC=nvcc

ARCH=-arch sm_20
CUDAFLAGS=-lineinfo $(ARCH)

all: TimingExample

TimingExample: TimingExample.cu 
	$(NVCC) TimingExample.cu -o TimingExample $(CUDAFLAGS) 

clean:
	rm -f *.o TimingExample.linkinfo TimingExample

