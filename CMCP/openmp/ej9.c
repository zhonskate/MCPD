#include <stdio.h>
#include <math.h>
#include <float.h>
#include <stdlib.h>
#include <omp.h>
#include "ctimer.h"
main(int argc, char**argv)
{
    ////// MULTIPLICACION ENTEROS//////
    double t1,t2,tucpu,tscpu;
    int bsize = 20;
    int asize = 20;
    int csize = asize+bsize;
    int a [asize];
    int b [bsize];
    int c [csize];

    srand(time(0));
    printf("/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/\n");
    printf("/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/\n");
    printf("Programa que calcula el Producto Matriz-Vector. \n");
    printf("------- \n");
    int i;
    // GENERACION DE DATOS //
    //
    for (i=0;i<bsize;i++){
        b[i]=(rand()%10);
    }
    for (i=0;i<asize;i++){
        a[i]=(rand()%10);
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

int * sumaVectores(int *m1, int *m2, int size1, int size2) {
    int sol[size1+size2];
    if (size1 > 3 || size2 > 3){
        int p1 = size1-1;
        int p2 = size2-2;
        int carry = 0;
        for(int i=size1+size2-1;i > 0; i--){
            if(p1 >=0 && p2 >=0){
                sol[i] = (m1[p1]+m2[p2]+carry)%10;
                carry = (int)(m1[p1]+m2[p2]+carry)/10
            }
            else if(p1>=0){
                sol[i] = (m1[p1]+carry)%10;
                carry = (int)(m1[p1]+carry)/10
            }
            else if(p2>=0){
                sol[i] = (m2[p2]+carry)%10;
                carry = (int)(m2[p2]+carry)/10
            }
            else {
                sol[i] = carry%10;
                carry = (int)carry/10;
            }
            p1--;
            p2--;
        }
        return sol;
    }
    else{
        mm1 = sumaVectores()
    }

}