
all: bandwidthTest

INCLUDE=-I/usr/local/cuda/samples/common/inc

bandwidthTest: bandwidthTest.cu
	nvcc -o bandwidthTest bandwidthTest.cu $(INCLUDE)

clean:
	rm -fr bandwidthTest
