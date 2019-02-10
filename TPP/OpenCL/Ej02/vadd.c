//------------------------------------------------------------------------------
//
// Name:       vadd.c
//
// Purpose:    Elementwise addition of two vectors (c = a + b)
//
// HISTORY:    Written by Tim Mattson, December 2009
//             Updated by Tom Deakin and Simon McIntosh-Smith, October 2012
//             Updated by Tom Deakin, July 2013
//             Updated by Tom Deakin, October 2014
//
//------------------------------------------------------------------------------


#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#include <unistd.h>
#else
#include <CL/cl.h>
#include <CL/cl_ext.h>
#endif

#include "cl_utils.h"

#define TOL    (0.001)   // tolerance used in floating point comparisons
#define LENGTH 1024*1024 // length of vectors a, b, and c

#define DEVICE_TYPE CL_DEVICE_TYPE_ALL

int main(int argc, char** argv)
{
    cl_int           err;           // error code returned from OpenCL calls
    cl_platform_id   platform;      // compute platform id
    cl_uint          numPlatforms;  // number of available platforms
    cl_device_id     devices[MAX_DEVICES];     // list of available devices
    cl_device_id     device;        // compute device id
    cl_context       context;       // compute context
    cl_command_queue commands;      // compute command queue
    cl_program       program;       // compute program
    cl_kernel        kernel;        // compute kernel

    cl_mem d_a;                     // device memory used for the input  a vector
    cl_mem d_b;                     // device memory used for the input  b vector
    cl_mem d_c;                     // device memory used for the output c vector

    char * filename="vadd.cl";      // name of the file containing the kernel

    float*       h_a = (float*) calloc(LENGTH, sizeof(float));       // a vector
    float*       h_b = (float*) calloc(LENGTH, sizeof(float));       // b vector
    float*       h_c = (float*) calloc(LENGTH, sizeof(float));       // c vector (a+b) returned from the compute device

    unsigned int correct;           // number of correct results

    // Fill vectors a and b with random float values
    size_t count = LENGTH;
    int i = 0;
    for(i = 0; i < LENGTH; i++){
        h_a[i] = rand() / (float)RAND_MAX;
        h_b[i] = rand() / (float)RAND_MAX;
    }

    // Obten la primera plataforma disponible
    err = clGetPlatformIDs(1, &platform, &numPlatforms);
    checkError(err, "Getting platforms");

    // Usa el primer dispositivo de tipo DEVICE_TYPE  de la plataforma
    err = clGetDeviceIDs(platform, DEVICE_TYPE, 1, &device, NULL);
    checkError(err, "Finding a device");

    // Crea un contexto simple con un solo dispositivo
    context = clCreateContext(0, 1, &device, NULL, NULL, &err);
    checkError(err, "Creating context");

    // Crea una cola de ordenes para alimentar el dispositivo:
    commands = clCreateCommandQueueWithProperties(context, device, 0, &err);
    checkError(err, "Creating command queue");

    // Construye el objeto programa
    char *kernelSource = getKernelSource(filename);
    program = clCreateProgramWithSource(context, 1, (const char **) &kernelSource, NULL, &err);

    // Compila el programa y construye la biblioteca dinamica de kernels
    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    checkError(err, "Building program");

    // Crea los objetos de memoria
    d_a  = clCreateBuffer(context,  CL_MEM_READ_ONLY,  LENGTH * sizeof(float), NULL, &err);
    checkError(err, "Creating buffer d_a");
    d_b  = clCreateBuffer(context,  CL_MEM_READ_ONLY,  LENGTH * sizeof(float), NULL, &err);
    checkError(err, "Creating buffer d_b");
    d_c  = clCreateBuffer(context,  CL_MEM_WRITE_ONLY, LENGTH * sizeof(float), NULL, &err);
    checkError(err, "Creating buffer d_c");

    // Crea el kernel a partir del programa
    kernel = clCreateKernel(program, "vadd", &err);
    checkError(err, "Creating kernel with vadd.cl");

    // Asocia objetos de memoria con los argumetnos de la funcion kernel
    err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_a);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_b);
    err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_c);
    err |= clSetKernelArg(kernel, 3, sizeof(unsigned int), &count);
    checkError(err, "Setting kernel arguments");

    // Escribe los datos de entrada del host al dispositivo
    err = clEnqueueWriteBuffer(commands, d_a, CL_FALSE, 0, LENGTH * sizeof(float), h_a, 0, NULL, NULL);
    checkError(err, "Copying h_a to device at d_a");
    err = clEnqueueWriteBuffer(commands, d_b, CL_FALSE, 0, LENGTH * sizeof(float), h_b, 0, NULL, NULL);
    checkError(err, "Copying h_b to device at d_b");

    double rtime = wtime();

    // Ejecuta el kernel
    err = clEnqueueNDRangeKernel(commands, kernel, 1, NULL, &count, NULL, 0, NULL, NULL);
    checkError(err, "Enqueueing kernel");

    // Lee los resultados del dispositivo al host
    err = clEnqueueReadBuffer( commands, d_c, CL_TRUE, 0, LENGTH * sizeof(float), h_c, 0, NULL, NULL );  
    checkError(err, "Error: Failed to read output array");

    // Espera a que finalicen todas las tareas antes de parar el temporizador
    err = clFinish(commands);
    checkError(err, "Waiting for kernel to finish");
   
    rtime = wtime() - rtime;
    printf("\nThe kernel ran in %lf seconds\n",rtime);

    // Comprueba los resultados
    correct = 0;
    float tmp;

    for(i = 0; i < LENGTH; i++)
    {
        tmp = h_a[i] + h_b[i];     // assign element i of a+b to tmp
        tmp -= h_c[i];             // compute deviation of expected and output result
        if(tmp*tmp < TOL*TOL)      // correct if square deviation is less than tolerance squared
            correct++;
        else {
            printf(" tmp %f h_a %f h_b %f h_c %f \n",tmp, h_a[i], h_b[i], h_c[i]);
        }
    }

    // Resume los resultados
    printf("C = A+B:  %d out of %d results were correct.\n", correct, LENGTH);

    // Libera los recursos y finaliza
    clReleaseMemObject(d_a);
    clReleaseMemObject(d_b);
    clReleaseMemObject(d_c);
    clReleaseProgram(program);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(commands);
    clReleaseContext(context);

    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}

