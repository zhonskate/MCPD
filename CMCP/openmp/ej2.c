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
    const long int M= 1048576;
    double TALLA = 8;
    double *b;
    b=malloc(M*sizeof(double));
    double suma;
    srand(time(0));
    printf("/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/\n");
    printf("/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/\n");
    printf("Programa que calcula el Producto Matriz-Vector. \n");
    printf("------- \n");
    int i;
    // GENERACION DE DATOS //
    //
    for (i=0;i<M;i++){
        b[i]=rand();
    }

    // PRODUCTO MATRIZ-VECTOR SECUENCIAL //
    //
    //
    printf("Voy a empezar la suma secuencial. \n");
    printf(" ------- \n"); 
    double alfa;
    ctimer(&t1,&tucpu,&tscpu);
    alfa=0.0;
    for(i=0;i<M;i++){
        alfa+=b[i];
    }
    ctimer(&t2,&tucpu,&tscpu);
    printf("Suma = %f \n",alfa);
    printf(" ------- \n");
    printf("Tiempo %f segundos \n",(float) (t2-t1));
    printf(" ------- \n");

    // PRODUCTO MATRIZ // PRODUCTO MATRIZ-VECTOR PARALELO / VECTOR PARALELO //
    printf("Empiezo la suma paralela\n");
    printf(" ------- \n");
    ctimer(&t1,&tucpu,&tscpu);
    int tb;
    int tid;
    tb=M/TALLA;
    omp_set_num_threads(TALLA);
    double sol;
    sol = 0.0;
    #pragma omp parallel for reduction(+:sol) private(i,tb)
    for (i=0;i<M;i++){
        sol += b[i];
    }
    ctimer(&t2,&tucpu,&tscpu);

    // SALIDA DE RESULTADOS //
    printf("Ha terminado la suma paralela\n");
    printf(" ------- \n");
    printf("Suma = %f \n",sol);
    printf(" ------- \n");
    // Fin del calculo del Producto Matriz-Vector paralelo
    printf("Tiempo %f segundos \n",(float) (t2-t1));
    printf("He acabado. \n");
    printf("/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/\n");
    printf("/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/\n");
}