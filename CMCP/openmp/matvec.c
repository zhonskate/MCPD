#include <stdio.h>
#include <math.h>
#include <float.h>
#include <stdlib.h>
#include <omp.h>
#include "ctimer.h"
main(int argc, char**argv)
{
    ////// PRODUCTO MATRIZ-VECTOR x=A*b //////
    // DECLARACION DE VARIABLES // // DECLARACION DE VARIABLES //
    double t1,t2,tucpu,tscpu;
    const int TALLA= 4;
    const long int M= 6400;
    const long int N= 3200;
    int i;
    int j;
    double *A;
    double *b;
    double *x;
    double *z;
    A=malloc(M*N*sizeof(double));
    b=malloc(M*sizeof(double));
    x=malloc(M*sizeof(double));
    z=malloc(M*sizeof(double));
    double suma;
    srand(time(0));
    printf("/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/\n");
    printf("/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/\n");
    printf("Programa que calcula el Producto Matriz-Vector. \n");
    printf("------- \n");

    // GENERACION DE DATOS //
    //
    for (i=0;i<M;i++){
        for (j=0;j<N;j++)
            A[i+j*M]=rand() % TALLA;
        b[i]=rand() % TALLA;
    }

    // PRODUCTO MATRIZ-VECTOR SECUENCIAL //
    //
    //
    printf("Voy a empezar el Producto Matriz-Vector secuencial. \n");
    printf(" ------- \n"); 
    double alfa;
    ctimer(&t1,&tucpu,&tscpu);
    for(i=0;i<M;i++){
        alfa=0.0;
        for (j=0;j<N;j++)
            alfa=alfa+A[i+j*M]*b[j];
        z[i]=alfa;
    }
    ctimer(&t2,&tucpu,&tscpu);
    printf("Producto Producto MatxVec MatxVec secuencial secuencial z(5) %f \n",z[5]);
    printf(" ------- \n");
    printf("Tiempo %f segundos \n",(float) (t2-t1));
    printf(" ------- \n");

    // PRODUCTO MATRIZ // PRODUCTO MATRIZ-VECTOR PARALELO / VECTOR PARALELO //
    printf("Empiezo el producto Matriz-Vector paralelo\n");
    printf(" ------- \n");
    ctimer(&t1,&tucpu,&tscpu);
    int tb;
    int tid;
    tb=M/TALLA;
    omp_set_num_threads(TALLA);
    #pragma omp parallel private(tid,i,j,alfa)
    {
        tid = omp_get_thread_num();
        for (i=0;i<tb;i++){
            alfa=0;
            for (j=0;j<N;j++)
                alfa=alfa+A[i+tb*tid+j*M]*b[j];
            x[i+tb*tid]=alfa;
        }
    }
    ctimer(&t2,&tucpu,&tscpu);

    // SALIDA DE RESULTADOS //
    printf("He terminado el Producto MatxVec paralelo \n");
    printf(" ------- \n");
    printf("Producto MatxVec paralelo x(5)= %f \n",x[5]);
    printf(" ------- \n");
    // Fin del calculo del Producto Matriz-Vector paralelo
    printf("Tiempo %f segundos \n",(float) (t2-t1));
    printf(" ------- \n");
    printf(" Comprobación \n");
    suma=0.0;
    for (i=1;i<M;i++)
        suma=suma + (x[i]-z[i])*(x[i]-z[i]);
    suma=sqrt(suma);
    printf("norma dif SEQ-PAR=%f\n",suma);
    printf(" ------- \n");
    printf("He acabado. \n");
    printf("/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/\n");
    printf("/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/\n");
}