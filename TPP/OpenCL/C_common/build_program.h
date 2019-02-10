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
    fread(source, sizeof(char), len, file);
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
        return EXIT_FAILURE;
    }

    return program;
}
