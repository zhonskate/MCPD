#include <stdlib.h>
#include <stdio.h>
#include <sys/types.h>
#include <unistd.h>
#include <fcntl.h>
#include <xmmintrin.h>



typedef float FLOAT;
//typedef float FLOAT;

// Cycle Counter Code
//
// Can be replaced with ippGetCpuFreqMhz and ippGetCpuClocks
// when IPP core functions are available.
//
typedef unsigned int UINT32;
typedef unsigned long long int UINT64;
typedef unsigned char UINT8;





// PPM Edge Enhancement Code
//
UINT8 header[22];
UINT8 R[76800];
UINT8 G[76800];
UINT8 B[76800];
UINT8 convR[76800];
UINT8 convG[76800];
UINT8 convB[76800];

#define K 4.0

FLOAT PSF[9] = {-K/8.0, -K/8.0, -K/8.0, -K/8.0, K+1.0, -K/8.0, -K/8.0, -K/8.0, -K/8.0};

int main(int argc, char *argv[])
{
    int fdin, fdout, bytesRead=0, bytesLeft, i, j,vuelta;
    double elapsed0, ucpu0, scpu0;
    double elapsed1, ucpu1, scpu1;
    float conversion [4];


  

    if(argc < 3)
    {
       printf("Usage: sharpen <file.ppm> <iterations> \n");
       exit(-1);
    }
    else
    {


        if((fdin = open(argv[1], O_RDONLY, 0644)) < 0)
        {
            printf("Error opening %s\n", argv[1]);
        }


        if((fdout = open("sharpen.ppm", (O_RDWR | O_CREAT), 0666)) < 0)
        {
            printf("Error opening %s\n", argv[1]);
        }

    }

    bytesLeft=21;

    //printf("Reading header\n");

    do
    {
        //printf("bytesRead=%d, bytesLeft=%d\n", bytesRead, bytesLeft);
        bytesRead=read(fdin, (void *)header, bytesLeft);
        bytesLeft -= bytesRead;
    } while(bytesLeft > 0);

    header[21]='\0';

    //printf("header = %s\n", header); 

    // Read RGB data
    for(i=0; i<76800; i++)
    {
        read(fdin, (void *)&R[i], 1); convR[i]=R[i];
        read(fdin, (void *)&G[i], 1); convG[i]=G[i];
        read(fdin, (void *)&B[i], 1); convB[i]=B[i];
    }
    __m128 temp;
    int z;
    //float *pResult = (float*) _aligned_malloc(318 * sizeof(float), 16);
    //__m128 *pResultSSE = (__m128*) pResult;

    // Start of convolution time stamp
    ctimer_(&elapsed0, &ucpu0, &scpu0);
    for (vuelta=1;vuelta<atoi(argv[2]);vuelta++)
    {

        // Skip first and last row, no neighbors to convolve with
        for(i=1; i<239; i++)
        {

            // Skip first and last column, no neighbors to convolve with
            //const int SSELength = 318 / 4;

            for(j=1; j<319; j=j+4)
            {
//#define TIME_SSE
//#ifdef TIME_SSE

                temp = _mm_set_ps(
                    (PSF[0] * (FLOAT)R[((i-1)*320)+j-1]), 
                    (PSF[0] * (FLOAT)R[((i-1)*320)+j+1-1]), 
                    (PSF[0] * (FLOAT)R[((i-1)*320)+j+2-1]),
                    (PSF[0] * (FLOAT)R[((i-1)*320)+j+3-1]));

                temp= _mm_add_ps(temp, _mm_set_ps(
                    (PSF[1] * (FLOAT)R[((i-1)*320)+j]), 
                    (PSF[1] * (FLOAT)R[((i-1)*320)+j+1]), 
                    (PSF[1] * (FLOAT)R[((i-1)*320)+j+2]),
                    (PSF[1] * (FLOAT)R[((i-1)*320)+j+3])));

                temp = _mm_add_ps(temp, _mm_set_ps(
                    (PSF[2] * (FLOAT)R[((i-1)*320)+j+1]), 
                    (PSF[2] * (FLOAT)R[((i-1)*320)+j+1+1]), 
                    (PSF[2] * (FLOAT)R[((i-1)*320)+j+1+2]),
                    (PSF[2] * (FLOAT)R[((i-1)*320)+j+1+3])));

                temp = _mm_add_ps(temp, _mm_set_ps(
                    (PSF[3] * (FLOAT)R[((i)*320)+j-1]), 
                    (PSF[3] * (FLOAT)R[((i)*320)+j-1+1]), 
                    (PSF[3] * (FLOAT)R[((i)*320)+j-1+2]),
                    (PSF[3] * (FLOAT)R[((i)*320)+j-1+3])));
                
                temp = _mm_add_ps(temp, _mm_set_ps(
                    (PSF[4] * (FLOAT)R[((i)*320)+j]), 
                    (PSF[4] * (FLOAT)R[((i)*320)+j+1]), 
                    (PSF[4] * (FLOAT)R[((i)*320)+j+2]),
                    (PSF[4] * (FLOAT)R[((i)*320)+j+3])));
                
                temp = _mm_add_ps(temp, _mm_set_ps(
                    (PSF[5] * (FLOAT)R[((i)*320)+j+1]), 
                    (PSF[5] * (FLOAT)R[((i)*320)+j+1+1]), 
                    (PSF[5] * (FLOAT)R[((i)*320)+j+1+2]),
                    (PSF[5] * (FLOAT)R[((i)*320)+j+1+3])));

                temp = _mm_add_ps(temp, _mm_set_ps(
                    (PSF[6] * (FLOAT)R[((i+1)*320)+j-1]), 
                    (PSF[6] * (FLOAT)R[((i+1)*320)+j-1+1]), 
                    (PSF[6] * (FLOAT)R[((i+1)*320)+j-1+2]),
                    (PSF[6] * (FLOAT)R[((i+1)*320)+j-1+3])));

                temp = _mm_add_ps(temp, _mm_set_ps(
                    (PSF[7] * (FLOAT)R[((i+1)*320)+j]), 
                    (PSF[7] * (FLOAT)R[((i+1)*320)+j+1]), 
                    (PSF[7] * (FLOAT)R[((i+1)*320)+j+2]),
                    (PSF[7] * (FLOAT)R[((i+1)*320)+j+3])));
                
                temp = _mm_add_ps(temp, _mm_set_ps(
                    (PSF[8] * (FLOAT)R[((i+1)*320)+j+1]), 
                    (PSF[8] * (FLOAT)R[((i+1)*320)+j+1+1]), 
                    (PSF[8] * (FLOAT)R[((i+1)*320)+j+1+2]),
                    (PSF[8] * (FLOAT)R[((i+1)*320)+j+1+3])));
                
                _mm_storeu_ps (conversion, temp);

                //if(j==1 && i==1 && vuelta ==1){printf ("%f\n",conversion[0]);}

                for (z=0;z<4;z++){
                    if(conversion[z]<0.0) conversion[z]=0.0;
                    if(conversion[z]>255.0) conversion[z]=255.0;
                    convR[(i*320)+j+(3-z)]=(UINT8)conversion[z];
                }

                temp = _mm_set_ps(
                    (PSF[0] * (FLOAT)G[((i-1)*320)+j-1]), 
                    (PSF[0] * (FLOAT)G[((i-1)*320)+j+1-1]), 
                    (PSF[0] * (FLOAT)G[((i-1)*320)+j+2-1]),
                    (PSF[0] * (FLOAT)G[((i-1)*320)+j+3-1]));

                temp= _mm_add_ps(temp, _mm_set_ps(
                    (PSF[1] * (FLOAT)G[((i-1)*320)+j]), 
                    (PSF[1] * (FLOAT)G[((i-1)*320)+j+1]), 
                    (PSF[1] * (FLOAT)G[((i-1)*320)+j+2]),
                    (PSF[1] * (FLOAT)G[((i-1)*320)+j+3])));

                temp = _mm_add_ps(temp, _mm_set_ps(
                    (PSF[2] * (FLOAT)G[((i-1)*320)+j+1]), 
                    (PSF[2] * (FLOAT)G[((i-1)*320)+j+1+1]), 
                    (PSF[2] * (FLOAT)G[((i-1)*320)+j+1+2]),
                    (PSF[2] * (FLOAT)G[((i-1)*320)+j+1+3])));

                temp = _mm_add_ps(temp, _mm_set_ps(
                    (PSF[3] * (FLOAT)G[((i)*320)+j-1]), 
                    (PSF[3] * (FLOAT)G[((i)*320)+j-1+1]), 
                    (PSF[3] * (FLOAT)G[((i)*320)+j-1+2]),
                    (PSF[3] * (FLOAT)G[((i)*320)+j-1+3])));
                
                temp = _mm_add_ps(temp, _mm_set_ps(
                    (PSF[4] * (FLOAT)G[((i)*320)+j]), 
                    (PSF[4] * (FLOAT)G[((i)*320)+j+1]), 
                    (PSF[4] * (FLOAT)G[((i)*320)+j+2]),
                    (PSF[4] * (FLOAT)G[((i)*320)+j+3])));
                
                temp = _mm_add_ps(temp, _mm_set_ps(
                    (PSF[5] * (FLOAT)G[((i)*320)+j+1]), 
                    (PSF[5] * (FLOAT)G[((i)*320)+j+1+1]), 
                    (PSF[5] * (FLOAT)G[((i)*320)+j+1+2]),
                    (PSF[5] * (FLOAT)G[((i)*320)+j+1+3])));

                temp = _mm_add_ps(temp, _mm_set_ps(
                    (PSF[6] * (FLOAT)G[((i+1)*320)+j-1]), 
                    (PSF[6] * (FLOAT)G[((i+1)*320)+j-1+1]), 
                    (PSF[6] * (FLOAT)G[((i+1)*320)+j-1+2]),
                    (PSF[6] * (FLOAT)G[((i+1)*320)+j-1+3])));

                temp = _mm_add_ps(temp, _mm_set_ps(
                    (PSF[7] * (FLOAT)G[((i+1)*320)+j]), 
                    (PSF[7] * (FLOAT)G[((i+1)*320)+j+1]), 
                    (PSF[7] * (FLOAT)G[((i+1)*320)+j+2]),
                    (PSF[7] * (FLOAT)G[((i+1)*320)+j+3])));
                
                temp = _mm_add_ps(temp, _mm_set_ps(
                    (PSF[8] * (FLOAT)G[((i+1)*320)+j+1]), 
                    (PSF[8] * (FLOAT)G[((i+1)*320)+j+1+1]), 
                    (PSF[8] * (FLOAT)G[((i+1)*320)+j+1+2]),
                    (PSF[8] * (FLOAT)G[((i+1)*320)+j+1+3])));
                
                _mm_storeu_ps (conversion, temp);

                //if(j==1 && i==1 && vuelta ==1){printf ("%f\n",conversion[0]);}

                for (z=0;z<4;z++){
                    if(conversion[z]<0.0) conversion[z]=0.0;
                    if(conversion[z]>255.0) conversion[z]=255.0;
                    convG[(i*320)+j+(3-z)]=(UINT8)conversion[z];
                }

                temp = _mm_set_ps(
                    (PSF[0] * (FLOAT)B[((i-1)*320)+j-1]), 
                    (PSF[0] * (FLOAT)B[((i-1)*320)+j+1-1]), 
                    (PSF[0] * (FLOAT)B[((i-1)*320)+j+2-1]),
                    (PSF[0] * (FLOAT)B[((i-1)*320)+j+3-1]));

                temp= _mm_add_ps(temp, _mm_set_ps(
                    (PSF[1] * (FLOAT)B[((i-1)*320)+j]), 
                    (PSF[1] * (FLOAT)B[((i-1)*320)+j+1]), 
                    (PSF[1] * (FLOAT)B[((i-1)*320)+j+2]),
                    (PSF[1] * (FLOAT)B[((i-1)*320)+j+3])));

                temp = _mm_add_ps(temp, _mm_set_ps(
                    (PSF[2] * (FLOAT)B[((i-1)*320)+j+1]), 
                    (PSF[2] * (FLOAT)B[((i-1)*320)+j+1+1]), 
                    (PSF[2] * (FLOAT)B[((i-1)*320)+j+1+2]),
                    (PSF[2] * (FLOAT)B[((i-1)*320)+j+1+3])));

                temp = _mm_add_ps(temp, _mm_set_ps(
                    (PSF[3] * (FLOAT)B[((i)*320)+j-1]), 
                    (PSF[3] * (FLOAT)B[((i)*320)+j-1+1]), 
                    (PSF[3] * (FLOAT)B[((i)*320)+j-1+2]),
                    (PSF[3] * (FLOAT)B[((i)*320)+j-1+3])));
                
                temp = _mm_add_ps(temp, _mm_set_ps(
                    (PSF[4] * (FLOAT)B[((i)*320)+j]), 
                    (PSF[4] * (FLOAT)B[((i)*320)+j+1]), 
                    (PSF[4] * (FLOAT)B[((i)*320)+j+2]),
                    (PSF[4] * (FLOAT)B[((i)*320)+j+3])));
                
                temp = _mm_add_ps(temp, _mm_set_ps(
                    (PSF[5] * (FLOAT)B[((i)*320)+j+1]), 
                    (PSF[5] * (FLOAT)B[((i)*320)+j+1+1]), 
                    (PSF[5] * (FLOAT)B[((i)*320)+j+1+2]),
                    (PSF[5] * (FLOAT)B[((i)*320)+j+1+3])));

                temp = _mm_add_ps(temp, _mm_set_ps(
                    (PSF[6] * (FLOAT)B[((i+1)*320)+j-1]), 
                    (PSF[6] * (FLOAT)B[((i+1)*320)+j-1+1]), 
                    (PSF[6] * (FLOAT)B[((i+1)*320)+j-1+2]),
                    (PSF[6] * (FLOAT)B[((i+1)*320)+j-1+3])));

                temp = _mm_add_ps(temp, _mm_set_ps(
                    (PSF[7] * (FLOAT)B[((i+1)*320)+j]), 
                    (PSF[7] * (FLOAT)B[((i+1)*320)+j+1]), 
                    (PSF[7] * (FLOAT)B[((i+1)*320)+j+2]),
                    (PSF[7] * (FLOAT)B[((i+1)*320)+j+3])));
                
                temp = _mm_add_ps(temp, _mm_set_ps(
                    (PSF[8] * (FLOAT)B[((i+1)*320)+j+1]), 
                    (PSF[8] * (FLOAT)B[((i+1)*320)+j+1+1]), 
                    (PSF[8] * (FLOAT)B[((i+1)*320)+j+1+2]),
                    (PSF[8] * (FLOAT)B[((i+1)*320)+j+1+3])));
                
                _mm_storeu_ps (conversion, temp);

                //if(j==1 && i==1 && vuelta ==1){printf ("%f\n",conversion[0]);}

                for (z=0;z<4;z++){
                    if(conversion[z]<0.0) conversion[z]=0.0;
                    if(conversion[z]>255.0) conversion[z]=255.0;
                    convB[(i*320)+j+(3-z)]=(UINT8)conversion[z];
                }
                
//#endif	// USE_DIVISION_METHOD
//#endif	// TIME_SSE

                /*temp=0;
                temp += (PSF[0] * (FLOAT)R[((i-1)*320)+j-1]);
                temp += (PSF[1] * (FLOAT)R[((i-1)*320)+j]);
                temp += (PSF[2] * (FLOAT)R[((i-1)*320)+j+1]);
                temp += (PSF[3] * (FLOAT)R[((i)*320)+j-1]);
                temp += (PSF[4] * (FLOAT)R[((i)*320)+j]);
                temp += (PSF[5] * (FLOAT)R[((i)*320)+j+1]);
                temp += (PSF[6] * (FLOAT)R[((i+1)*320)+j-1]);
                temp += (PSF[7] * (FLOAT)R[((i+1)*320)+j]);
                temp += (PSF[8] * (FLOAT)R[((i+1)*320)+j+1]);
                if(temp<0.0) temp=0.0;
                if(temp>255.0) temp=255.0;
                convR[(i*320)+j]=(UINT8)temp;

                temp=0;
                temp += (PSF[0] * (FLOAT)G[((i-1)*320)+j-1]);
                temp += (PSF[1] * (FLOAT)G[((i-1)*320)+j]);
                temp += (PSF[2] * (FLOAT)G[((i-1)*320)+j+1]);
                temp += (PSF[3] * (FLOAT)G[((i)*320)+j-1]);
                temp += (PSF[4] * (FLOAT)G[((i)*320)+j]);
                temp += (PSF[5] * (FLOAT)G[((i)*320)+j+1]);
                temp += (PSF[6] * (FLOAT)G[((i+1)*320)+j-1]);
                temp += (PSF[7] * (FLOAT)G[((i+1)*320)+j]);
                temp += (PSF[8] * (FLOAT)G[((i+1)*320)+j+1]);
                if(temp<0.0) temp=0.0;
                if(temp>255.0) temp=255.0;
                convG[(i*320)+j]=(UINT8)temp;

                temp=0;
                temp += (PSF[0] * (FLOAT)B[((i-1)*320)+j-1]);
                temp += (PSF[1] * (FLOAT)B[((i-1)*320)+j]);
                temp += (PSF[2] * (FLOAT)B[((i-1)*320)+j+1]);
                temp += (PSF[3] * (FLOAT)B[((i)*320)+j-1]);
                temp += (PSF[4] * (FLOAT)B[((i)*320)+j]);
                temp += (PSF[5] * (FLOAT)B[((i)*320)+j+1]);
                temp += (PSF[6] * (FLOAT)B[((i+1)*320)+j-1]);
                temp += (PSF[7] * (FLOAT)B[((i+1)*320)+j]);
                temp += (PSF[8] * (FLOAT)B[((i+1)*320)+j+1]);
                if(temp<0.0) temp=0.0;
                if(temp>255.0) temp=255.0;
                convB[(i*320)+j]=(UINT8)temp;*/
            }
        }
}

    // End of convolution time stamp
 ctimer_(&elapsed1, &ucpu1, &scpu1);
    printf("iteraciones \t %d \t Tiempo: \t %fs \t (real) \t %fs \t (cpu) \t %fs \t(sys)\n", 
                 atoi(argv[2]), elapsed1-elapsed0, ucpu1-ucpu0, scpu1-scpu0);

    write(fdout, (void *)header, 21);

    // Write RGB data
    for(i=0; i<76800; i++)
    {
        write(fdout, (void *)&convR[i], 1);
        write(fdout, (void *)&convG[i], 1);
        write(fdout, (void *)&convB[i], 1);
    }


    close(fdin);
    close(fdout);
 
}
