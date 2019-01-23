
all: deviceQuery

INCLUDE=-I/usr/local/cuda/samples/common/inc

deviceQuery: deviceQuery.cpp
	nvcc -o deviceQuery deviceQuery.cpp $(INCLUDE)

clean:
	rm -fr deviceQuery
