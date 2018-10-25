#include <stdio.h>
#include <math.h>
#include <float.h>
#include <stdlib.h>
#include <omp.h>
#include "ctimer.h"
main(int argc, char**argv)
{
    double t1,t2,tucpu,tscpu;
    const long int M= 1000000;
    double *b;
    b=malloc(M*sizeof(double));
    double suma;
    srand(time(0));
    printf("/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/\n");
    printf("/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/\n");
    printf("Programa que calcula el Mayor valor. \n");
    printf("------- \n");
    int i;
    // GENERACION DE DATOS //
    //
    for (i=0;i<M;i++){
        b[i]=rand();
    }

    printf("Secuencial. \n");
    printf(" ------- \n"); 
    double alfa;
    ctimer(&t1,&tucpu,&tscpu);
    alfa=0.0;
    for(i=0;i<M;i++){
        if (b[i]>alfa){
            alfa=b[i];
        }
    }
    ctimer(&t2,&tucpu,&tscpu);
    printf("Mayor = %f \n",alfa);
    printf(" ------- \n");
    printf("Tiempo %f segundos \n",(float) (t2-t1));
    printf(" ------- \n");

    // PRODUCTO MATRIZ // PRODUCTO MATRIZ-VECTOR PARALELO / VECTOR PARALELO //
    printf("paralelo\n");
    ctimer(&t1,&tucpu,&tscpu);
    int tb;
    int tid;
    int nthreads = 4;
    tb=(int)M/nthreads;
    omp_set_num_threads(nthreads);
    double privatesol;
    privatesol = 0.0;
    double *sol;
    sol=malloc(nthreads*sizeof(double));
    #pragma omp parallel private(i,tid,privatesol) 
    {
        i=0;
        tid = omp_get_thread_num();
        printf("thread = %d i = %d\n",tid,i);
        printf("nthreads = %d tb = %d\n",nthreads,tb);
        for (i=0;i<tb;i++){
            // printf("i = %d b = %d\n",i,tid*tb + i);
            if (privatesol < b[tb*tid + i]){
                privatesol = b[tb*tid + i];
            } 
        }
        sol[tid]=privatesol;
        for (i=0;i<log2(nthreads);i++){
            if((tid+nthreads)%(int)pow(2,i+1)==0){
                if(sol[tid+(int)pow(2,i)]> sol[tid]){
                    sol[tid] = sol[tid+(int)pow(2,i)];
                }
            }
        }

    }
    ctimer(&t2,&tucpu,&tscpu);

    // SALIDA DE RESULTADOS //
    //printf("mayor = %f \n",privatesol);
    printf(" ------- \n");
    printf("mayor = %f \n",sol[0]);
    // Fin del calculo del Producto Matriz-Vector paralelo
    printf("Tiempo %f segundos \n",(float) (t2-t1));
    printf("He acabado. \n");
    free(b);
    free(sol);
    printf("/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/\n");
    printf("/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/\n");
}