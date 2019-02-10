#include <stdio.h>
#include <stdlib.h>
#include <string.h>
//#include <err_code.h>
#include "cl_utils.h"

#ifdef __cplusplus
 #include <cstdio>
#endif

#if defined(__APPLE__) || defined(__MACOSX)
#include <OpenCL/opencl.h>
#else
//#include <CL/cl.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#else
#include <sys/time.h>
#endif

double wtime()
{
#ifdef _OPENMP
   /* Use omp_get_wtime() if we can */
   return omp_get_wtime();
#else
   /* Use a generic timer */
   static int sec = -1;
   struct timeval tv;
   gettimeofday(&tv, NULL);
   if (sec < 0) sec = tv.tv_sec;
   return (tv.tv_sec - sec) + 1.0e-6*tv.tv_usec;
#endif
}


//#pragma once
/*----------------------------------------------------------------------------
 *
 * Name:     err_code()
 *
 * Purpose:  Function to output descriptions of errors for an input error code
 *           and quit a program on an error with a user message
 *
 *
 * RETURN:   echoes the input error code / echos user message and exits
 *
 * HISTORY:  Written by Tim Mattson, June 2010
 *           This version automatically produced by genErrCode.py
 *           script written by Tom Deakin, August 2013
 *           Modified by Bruce Merry, March 2014
 *           Updated by Tom Deakin, October 2014
 *               Included the checkError function written by
 *               James Price and Simon McIntosh-Smith
 *
 *----------------------------------------------------------------------------
 */

const char *err_code (cl_int err_in)
{
    switch (err_in) {
        case CL_SUCCESS:
            return (char*)"CL_SUCCESS";
        case CL_DEVICE_NOT_FOUND:
            return (char*)"CL_DEVICE_NOT_FOUND";
        case CL_DEVICE_NOT_AVAILABLE:
            return (char*)"CL_DEVICE_NOT_AVAILABLE";
        case CL_COMPILER_NOT_AVAILABLE:
            return (char*)"CL_COMPILER_NOT_AVAILABLE";
        case CL_MEM_OBJECT_ALLOCATION_FAILURE:
            return (char*)"CL_MEM_OBJECT_ALLOCATION_FAILURE";
        case CL_OUT_OF_RESOURCES:
            return (char*)"CL_OUT_OF_RESOURCES";
        case CL_OUT_OF_HOST_MEMORY:
            return (char*)"CL_OUT_OF_HOST_MEMORY";
        case CL_PROFILING_INFO_NOT_AVAILABLE:
            return (char*)"CL_PROFILING_INFO_NOT_AVAILABLE";
        case CL_MEM_COPY_OVERLAP:
            return (char*)"CL_MEM_COPY_OVERLAP";
        case CL_IMAGE_FORMAT_MISMATCH:
            return (char*)"CL_IMAGE_FORMAT_MISMATCH";
        case CL_IMAGE_FORMAT_NOT_SUPPORTED:
            return (char*)"CL_IMAGE_FORMAT_NOT_SUPPORTED";
        case CL_BUILD_PROGRAM_FAILURE:
            return (char*)"CL_BUILD_PROGRAM_FAILURE";
        case CL_MAP_FAILURE:
            return (char*)"CL_MAP_FAILURE";
        case CL_MISALIGNED_SUB_BUFFER_OFFSET:
            return (char*)"CL_MISALIGNED_SUB_BUFFER_OFFSET";
        case CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST:
            return (char*)"CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
        case CL_INVALID_VALUE:
            return (char*)"CL_INVALID_VALUE";
        case CL_INVALID_DEVICE_TYPE:
            return (char*)"CL_INVALID_DEVICE_TYPE";
        case CL_INVALID_PLATFORM:
            return (char*)"CL_INVALID_PLATFORM";
        case CL_INVALID_DEVICE:
            return (char*)"CL_INVALID_DEVICE";
        case CL_INVALID_CONTEXT:
            return (char*)"CL_INVALID_CONTEXT";
        case CL_INVALID_QUEUE_PROPERTIES:
            return (char*)"CL_INVALID_QUEUE_PROPERTIES";
        case CL_INVALID_COMMAND_QUEUE:
            return (char*)"CL_INVALID_COMMAND_QUEUE";
        case CL_INVALID_HOST_PTR:
            return (char*)"CL_INVALID_HOST_PTR";
        case CL_INVALID_MEM_OBJECT:
            return (char*)"CL_INVALID_MEM_OBJECT";
        case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:
            return (char*)"CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
        case CL_INVALID_IMAGE_SIZE:
            return (char*)"CL_INVALID_IMAGE_SIZE";
        case CL_INVALID_SAMPLER:
            return (char*)"CL_INVALID_SAMPLER";
        case CL_INVALID_BINARY:
            return (char*)"CL_INVALID_BINARY";
        case CL_INVALID_BUILD_OPTIONS:
            return (char*)"CL_INVALID_BUILD_OPTIONS";
        case CL_INVALID_PROGRAM:
            return (char*)"CL_INVALID_PROGRAM";
        case CL_INVALID_PROGRAM_EXECUTABLE:
            return (char*)"CL_INVALID_PROGRAM_EXECUTABLE";
        case CL_INVALID_KERNEL_NAME:
            return (char*)"CL_INVALID_KERNEL_NAME";
        case CL_INVALID_KERNEL_DEFINITION:
            return (char*)"CL_INVALID_KERNEL_DEFINITION";
        case CL_INVALID_KERNEL:
            return (char*)"CL_INVALID_KERNEL";
        case CL_INVALID_ARG_INDEX:
            return (char*)"CL_INVALID_ARG_INDEX";
        case CL_INVALID_ARG_VALUE:
            return (char*)"CL_INVALID_ARG_VALUE";
        case CL_INVALID_ARG_SIZE:
            return (char*)"CL_INVALID_ARG_SIZE";
        case CL_INVALID_KERNEL_ARGS:
            return (char*)"CL_INVALID_KERNEL_ARGS";
        case CL_INVALID_WORK_DIMENSION:
            return (char*)"CL_INVALID_WORK_DIMENSION";
        case CL_INVALID_WORK_GROUP_SIZE:
            return (char*)"CL_INVALID_WORK_GROUP_SIZE";
        case CL_INVALID_WORK_ITEM_SIZE:
            return (char*)"CL_INVALID_WORK_ITEM_SIZE";
        case CL_INVALID_GLOBAL_OFFSET:
            return (char*)"CL_INVALID_GLOBAL_OFFSET";
        case CL_INVALID_EVENT_WAIT_LIST:
            return (char*)"CL_INVALID_EVENT_WAIT_LIST";
        case CL_INVALID_EVENT:
            return (char*)"CL_INVALID_EVENT";
        case CL_INVALID_OPERATION:
            return (char*)"CL_INVALID_OPERATION";
        case CL_INVALID_GL_OBJECT:
            return (char*)"CL_INVALID_GL_OBJECT";
        case CL_INVALID_BUFFER_SIZE:
            return (char*)"CL_INVALID_BUFFER_SIZE";
        case CL_INVALID_MIP_LEVEL:
            return (char*)"CL_INVALID_MIP_LEVEL";
        case CL_INVALID_GLOBAL_WORK_SIZE:
            return (char*)"CL_INVALID_GLOBAL_WORK_SIZE";
        case CL_INVALID_PROPERTY:
            return (char*)"CL_INVALID_PROPERTY";

        default:
            return (char*)"UNKNOWN ERROR";
    }
}


void check_error(cl_int err, const char *operation, char *filename, int line)
{
    if (err != CL_SUCCESS)
    {
        fprintf(stderr, "Error during operation '%s', ", operation);
        fprintf(stderr, "in '%s' on line %d\n", filename, line);
        fprintf(stderr, "Error code was \"%s\" (%d)\n", err_code(err), err);
        exit(EXIT_FAILURE);
    }
}

#define checkError(E, S) check_error(E,S,__FILE__,__LINE__)


/*------------------------------------------------------------------------------
 *
 * Name:       device_picker.h
 *
 * Purpose:    Provide a simple CLI to specify an OpenCL device at runtime
 *
 * Note:       Must be included AFTER the relevant OpenCL header
 *             See one of the Matrix Multiply exercises for usage
 *
 * HISTORY:    Method written by James Price, October 2014
 *             Extracted to a common header by Tom Deakin, November 2014
 */


#define MAX_PLATFORMS     8
#define MAX_DEVICES      16
#define MAX_INFO_STRING 256


unsigned getDeviceList(cl_device_id devices[MAX_DEVICES])
{
  cl_int err;

  // Get list of platforms
  cl_uint numPlatforms = 0;
  cl_platform_id platforms[MAX_PLATFORMS];
  err = clGetPlatformIDs(MAX_PLATFORMS, platforms, &numPlatforms);
  checkError(err, "getting platforms");

  // Enumerate devices
  unsigned numDevices = 0;
  for (int i = 0; i < numPlatforms; i++)
  {
    cl_uint num = 0;
    err = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL,
                         MAX_DEVICES-numDevices, devices+numDevices, &num);
    checkError(err, "getting deviceS");
    numDevices += num;
  }

  return numDevices;
}

void getDeviceName(cl_device_id device, char name[MAX_INFO_STRING])
{
  cl_device_info info = CL_DEVICE_NAME;

  // Special case for AMD
#ifdef CL_DEVICE_BOARD_NAME_AMD
  clGetDeviceInfo(device, CL_DEVICE_VENDOR, MAX_INFO_STRING, name, NULL);
  if (strstr(name, "Advanced Micro Devices"))
    info = CL_DEVICE_BOARD_NAME_AMD;
#endif

  clGetDeviceInfo(device, info, MAX_INFO_STRING, name, NULL);
}

/* Gets a given device (did) from a given platform (pid) */
unsigned getPlatformAndDevice(int pid, cl_platform_id *platform, int did, cl_device_id *device) {
  cl_int err;
  cl_uint numPlatforms = 0;
  cl_platform_id platforms[MAX_PLATFORMS];
  err = clGetPlatformIDs(MAX_PLATFORMS, platforms, &numPlatforms);
  checkError(err, "Finding platforms");
  if ( (numPlatforms == 0) || (pid >= numPlatforms) ) {
     printf("Platform %d unavailable. Found %d platform\n", pid, numPlatforms);
     return EXIT_FAILURE;
  }
  *platform = platforms[pid];

  cl_uint numDevices = 0;
  cl_device_id devices[MAX_DEVICES];
  err= clGetDeviceIDs(*platform, CL_DEVICE_TYPE_ALL, MAX_DEVICES, devices, &numDevices);
  checkError(err, "Finding devices");
  if ( (numDevices == 0) ||  (did >= numDevices) ) {
     printf("Device %d unavailable in platform %d. Found %d devices\n", did, pid, numDevices);
     return EXIT_FAILURE;
  }
  *device = devices[did];
  return EXIT_SUCCESS;
}


int parseUInt(const char *str, cl_uint *output)
{
  char *next;
  *output = strtoul(str, &next, 10);
  return !strlen(next);
}

void parseArguments(int argc, char *argv[], cl_uint *deviceIndex)
{
  for (int i = 1; i < argc; i++)
  {
    if (!strcmp(argv[i], "--list"))
    {
      // Get list of devices
      cl_device_id devices[MAX_DEVICES];
      unsigned numDevices = getDeviceList(devices);

      // Print device names
      if (numDevices == 0)
      {
        printf("No devices found.\n");
      }
      else
      {
        printf("\n");
        printf("Devices:\n");
        for (int i = 0; i < numDevices; i++)
        {
          char name[MAX_INFO_STRING];
          getDeviceName(devices[i], name);
          printf("%2d: %s\n", i, name);
        }
        printf("\n");
      }
      exit(0);
    }
    else if (!strcmp(argv[i], "--device"))
    {
      if (++i >= argc || !parseUInt(argv[i], deviceIndex))
      {
        printf("Invalid device index\n");
        exit(1);
      }
    }
    else if (!strcmp(argv[i], "--help") || !strcmp(argv[i], "-h"))
    {
      printf("\n");
      printf("Usage: ./program [OPTIONS]\n\n");
      printf("Options:\n");
      printf("  -h  --help               Print the message\n");
      printf("      --list               List available devices\n");
      printf("      --device     INDEX   Select device at INDEX\n");
      printf("\n");
      exit(0);
    }
  }
}


//------------------------------------------------------------------------------
//
// Name:     device_info()
//
// Purpose:  Function to output key parameters about the input OpenCL device.
//
//
// RETURN:   The OCL_SUCESS or the error code from one of the OCL function
//           calls internal to this function
//
// HISTORY:  Written by Tim Mattson, June 2010
//
//------------------------------------------------------------------------------
//
//  define VERBOSE if you want to print info about work groups sizes
//#define  VERBOSE 1
#ifdef VERBOSE
     extern int err_code(cl_int);
#endif

int output_device_info(cl_device_id device_id)
{
    int err;                            // error code returned from OpenCL calls
    cl_device_type device_type;         // Parameter defining the type of the compute device
    cl_uint comp_units;                 // the max number of compute units on a device
    cl_char vendor_name[1024] = {0};    // string to hold vendor name for compute device
    cl_char device_name[1024] = {0};    // string to hold name of compute device
#ifdef VERBOSE
    cl_uint          max_work_itm_dims;
    size_t           max_wrkgrp_size;
    size_t          *max_loc_size;
#endif


    err = clGetDeviceInfo(device_id, CL_DEVICE_NAME, sizeof(device_name), &device_name, NULL);
    checkError(err, "Error: Failed to access device name!\n");
    printf(" \n Device is  %s ",device_name);

    err = clGetDeviceInfo(device_id, CL_DEVICE_TYPE, sizeof(device_type), &device_type, NULL);
    checkError(err, "Error: Failed to access device type information!\n");
    if(device_type  == CL_DEVICE_TYPE_GPU)
       printf(" GPU from ");

    else if (device_type == CL_DEVICE_TYPE_CPU)
       printf("\n CPU from ");

    else 
       printf("\n non  CPU or GPU processor from ");

    err = clGetDeviceInfo(device_id, CL_DEVICE_VENDOR, sizeof(vendor_name), &vendor_name, NULL);
    checkError(err, "Error: Failed to access device vendor name!\n");
    printf(" %s ",vendor_name);

    err = clGetDeviceInfo(device_id, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &comp_units, NULL);
    checkError(err, "Error: Failed to access device number of compute units !\n");
    printf(" with a max of %d compute units \n",comp_units);

#ifdef VERBOSE
//
// Optionally print information about work group sizes
//
    err = clGetDeviceInfo( device_id, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(cl_uint), 
                               &max_work_itm_dims, NULL);
    checkError(err, "Error: Failed to get device Info (CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS)!\n",
                                                                            err_code(err));
    
    max_loc_size = (size_t*)malloc(max_work_itm_dims * sizeof(size_t));
    if(max_loc_size == NULL){
       printf(" malloc failed\n");
       return EXIT_FAILURE;
    }
    err = clGetDeviceInfo( device_id, CL_DEVICE_MAX_WORK_ITEM_SIZES, max_work_itm_dims* sizeof(size_t), 
                               max_loc_size, NULL);
    checkError(err, "Error: Failed to get device Info (CL_DEVICE_MAX_WORK_ITEM_SIZES)!\n",err_code(err));
    err = clGetDeviceInfo( device_id, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), 
                               &max_wrkgrp_size, NULL);
    checkError(err, "Error: Failed to get device Info (CL_DEVICE_MAX_WORK_GROUP_SIZE)!\n",err_code(err));
    printf("work group, work item information");
    printf("\n max loc dim ");
    for(int i=0; i< max_work_itm_dims; i++)
      printf(" %d ",(int)(*(max_loc_size+i)));
    printf("\n");
    printf(" Max work group size = %d\n",(int)max_wrkgrp_size);
#endif

    return CL_SUCCESS;
}


char * getKernelSource(const char *filename)
{   
    FILE *file = fopen(filename, "r");
    if (!file)
    {   
        fprintf(stderr, "Error: Could not open kernel source file\n");
        exit(EXIT_FAILURE);
    }
    fseek(file, 0, SEEK_END);
    int len = ftell(file) + 1;
    rewind(file);
    
    char *source = (char *)calloc(sizeof(char), len);
    if (!source)
    {   
        fprintf(stderr, "Error: Could not allocate memory for source string\n");
        exit(EXIT_FAILURE);
    }
    int size = fread(source, sizeof(char), len, file);
    fclose(file);
    return source;
}

cl_program buildProgram(cl_context context, cl_device_id device, const char *kernelName)
{
    cl_program program;
    cl_int     err;
    char *kernelSource;

    kernelSource = getKernelSource(kernelName);
    // Create the comput program from the source buffer
    program = clCreateProgramWithSource(context, 1, (const char **) &kernelSource, NULL, &err);
    checkError(err, "Creating program with vadd.cl");
    free(kernelSource);
    // Build the program
    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        size_t len;
        char buffer[2048];

        printf("Error: Failed to build program executable!\n%s\n", err_code(err));
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
        printf("%s\n", buffer);
        exit(EXIT_FAILURE);
    }

    return program;
}

void showKernelWorkGroupInfo(const char * kernelName, cl_kernel kernel, cl_device_id device) {
  size_t st[3];
  cl_int err;

  printf("WorkGroupInfo of kernel '%s':\n", kernelName);

  err =  clGetKernelWorkGroupInfo(kernel, device, CL_KERNEL_WORK_GROUP_SIZE,
           sizeof(size_t), st, NULL);
  checkError(err, "Error to get CL_KERNEL_WORK_GROUP_SIZE");
  printf("CL_KERNEL_WORK_GROUP_SIZE: %lu\n", st[0]);

  err =  clGetKernelWorkGroupInfo(kernel, device, CL_KERNEL_COMPILE_WORK_GROUP_SIZE,
           3*sizeof(size_t), st, NULL);
  checkError(err, "Error to get CL_KERNEL_COMPILE_WORK_GROUP_SIZE");
  printf("CL_KERNEL_COMPILE_WORK_GROUP_SIZE: (%lu, %lu, %lu) \n", st[0], st[1], st[2]);

  err =  clGetKernelWorkGroupInfo(kernel, device, CL_KERNEL_LOCAL_MEM_SIZE,
           sizeof(cl_ulong), st, NULL);
  checkError(err, "Error to get CL_KERNEL_LOCAL_MEM_SIZE");
  printf("CL_KERNEL_LOCAL_MEM_SIZE: %lu \n", st[0]);

  err =  clGetKernelWorkGroupInfo(kernel, device, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
           sizeof(size_t), st, NULL);
  checkError(err, "Error to get CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE");
  printf("CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE: %lu \n", st[0]);

  err =  clGetKernelWorkGroupInfo(kernel, device, CL_KERNEL_PRIVATE_MEM_SIZE,
           sizeof(cl_ulong), st, NULL);
  checkError(err, "Error to get CL_KERNEL_PRIVATE_MEM_SIZE");
  printf("CL_KERNEL_PRIVATE_MEM_SIZE: %lu \n", st[0]);

}

void eventProfiling(cl_event event) {
  cl_int err;
  size_t return_bytes;

  cl_ulong queuedTime;
  err = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_QUEUED,
          sizeof(cl_ulong), &queuedTime, &return_bytes);

  cl_ulong submittedTime;
  err |= clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_SUBMIT,
          sizeof(cl_ulong), &submittedTime, &return_bytes);

  cl_ulong startTime;
  err |= clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START,
          sizeof(cl_ulong), &startTime, &return_bytes);

  cl_ulong endTime;
  err |= clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END,
          sizeof(cl_ulong), &endTime, &return_bytes);
  checkError(err, "Profiling event");

  printf("\nProfiling information:\n");
  printf("Queued Time: %g ms.\n", (submittedTime - queuedTime)/1000000.0);
  printf("Wait Time: %g ms.\n", (startTime - queuedTime)/1000000.0);
  printf("Run Time: %g ms.\n\n", (endTime - startTime)/1000000.0);

}
