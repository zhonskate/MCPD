#include <stdio.h>
#include <stdlib.h>
#include <string.h>
//#include <err_code.h>

#ifdef __cplusplus
 #include <cstdio>
#endif

#if defined(__APPLE__) || defined(__MACOSX)
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

double wtime();

const char *err_code (cl_int err_in);

void check_error(cl_int err, const char *operation, char *filename, int line);

#define checkError(E, S) check_error(E,S,__FILE__,__LINE__)

#define MAX_PLATFORMS     8
#define MAX_DEVICES      16
#define MAX_INFO_STRING 256

unsigned getDeviceList(cl_device_id devices[MAX_DEVICES]);

unsigned getPlatformAndDevice(int, cl_platform_id *, int, cl_device_id *);

void getDeviceName(cl_device_id device, char name[MAX_INFO_STRING]);

int parseUInt(const char *str, cl_uint *output);

void parseArguments(int argc, char *argv[], cl_uint *deviceIndex);

/*
#ifdef VERBOSE
     extern int err_code(cl_int);
#endif
*/

int output_device_info(cl_device_id device_id);

char * getKernelSource(const char *filename);

cl_program buildProgram(cl_context context, cl_device_id device, const char *kernelName);

void showKernelWorkGroupInfo(const char *kernelName, cl_kernel kernel, cl_device_id device);

void eventProfiling(cl_event event);
