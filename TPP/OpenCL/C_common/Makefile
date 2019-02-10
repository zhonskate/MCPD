
ifndef CC
	CC = gcc
endif

include ../make.inc

# Change this variable to specify the device type
# to the OpenCL device type of choice. You can also
# edit the variable in the source.
ifndef DEVICE
	DEVICE = CL_DEVICE_TYPE_DEFAULT
endif

# Check our platform and make sure we define the APPLE variable
# and set up the right compiler flags and libraries
PLATFORM = $(shell uname -s)
ifeq ($(PLATFORM), Darwin)
	LIBS = -framework OpenCL
endif

CCFLAGS += -D DEVICE=$(DEVICE)

cl_utils.o: cl_utils.c
	$(CC) $(CCFLAGS) -I$(OPENCL_INC) -c $< -o $@

clean:
	rm -f *.o
