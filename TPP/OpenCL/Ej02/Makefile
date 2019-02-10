#
# Matrix Multiply example makefile
#
# History: Written  by Tim mattson, August 2010
#          Modified by Tom Deakin and Simon McIntosh-Smith, October 2012
#          Modified by Tom Deakin, July 2013
#          Modified by Tom Deakin, October 2014
#

ifndef CC
	CC = gcc
endif

include ../make.inc

COMMON_DIR = ../C_common

OBJS = cl_utils.o
EXEC = vadd


# Check our platform and make sure we define the APPLE variable
# and set up the right compiler flags and libraries
PLATFORM = $(shell uname -s)
ifeq ($(PLATFORM), Darwin)
	LIBS = -lm -framework OpenCL
endif


all: $(EXEC)

vadd: $(OBJS) vadd.c
	$(CC) $^ $(CCFLAGS) $(LIBS) -I$(COMMON_DIR) -I$(OPENCL_INC) -o $(EXEC)

cl_utils.o: $(COMMON_DIR)/cl_utils.c
	$(CC) -c $^ $(CCFLAGS) -I$(OPENCL_INC) -o $@

.c.o:
	$(CC) -c $< $(CCFLAGS) -I$(OPENCL_INC) -o $@


clean:
	rm -f $(OBJS) $(EXEC)
