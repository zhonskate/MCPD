#include <stdio.h>
#include <math.h>
#include <float.h>
#include <stdlib.h>
#include <omp.h>
#include "ctimer.h"

int * sumaVectores(int *m1, int *m2, int size1, int size2) {
    static int * sol = malloc(size1+size2 * sizeof(int));
    int i;
    for(i=0;i < 8; i++){
        sol[i] = 0;
    }
    int p1 = size1-1;
    int p2 = size2-1;
    int carry = 0;
    for(i=8-1;i >= 0; i--){
        if(p1 >=0 && p2 >=0){
            //printf("--1--\n");
            sol[i] = (m1[p1]+m2[p2]+carry)%10;
            carry = (int)(m1[p1]+m2[p2]+carry)/10;
        }
        else if(p1>=0){
            //printf("--2--\n");
            sol[i] = (m1[p1]+carry)%10;
            carry = (int)(m1[p1]+carry)/10;
        }
        else if(p2>=0){
            //printf("--3--\n");
            sol[i] = (m2[p2]+carry)%10;
            carry = (int)(m2[p2]+carry)/10;
        }
        else {
            //printf("--4--\n");
            sol[i] = carry%10;
            carry = (int)carry/10;
        }
        p1--;
        p2--;
    }
    return sol;
}

int * multVectorNum(int *m1, int size1, int num) {
    static int sol2[4];
    int i;
    for(i=0;i < 4; i++){
        sol2[i] = 0;
    }
    int p1 = size1-1;
    int carry = 0;
    for(i=3;i >= 0; i--){
        if(p1 >=0){
            sol2[i] = (m1[p1]*num+carry)%10;
            carry = (int)(m1[p1]*num+carry)/10;
        }
        else {
            sol2[i] = carry%10;
            carry = (int)carry/10;
        }
        //printf("carry %d \n", carry);
        p1--;
        //printf("p1 %d \n", p1);
    }
    return sol2;
}

int * addZeros(int *m1, int size1, int zeros) {
    static int sol3[5];
    int i;
    for(i=0;i < 8; i++){
        sol3[i] = 0;
    }
    for(i=0;i < size1; i++){
        sol3[i] = m1[i];
    }
    for(i=size1;i < size1+zeros; i++){
        sol3[i] = 0;
    }
    return sol3;
}


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
    /*alfa=0.0;
    for(i=0;i<M;i++){
        alfa+=b[i];
    }*/
    int ej1 [3];
    int ej2 [5];

    int loop;
    for(loop = 0; loop < 3; loop++)
        ej1[loop] = loop + 4;
    for(loop = 0; loop < 5; loop++)
        ej2[loop] = loop + 2;  
    
    for(loop = 0; loop < 3; loop++)
        printf("ej1 %d \n", ej1[loop]);

    printf(" ------- \n");

    for(loop = 0; loop < 5; loop++)
        printf("ej2 %d \n", ej2[loop]);

    printf(" ------- \n");

    int *solu;

    solu = sumaVectores(ej1,ej2,3,5);
    for(loop = 0; loop < 3+5; loop++)
        printf("sol %d \n", solu[loop]);

    printf(" ------- \n");
    
    int *solu2;

    solu2 = multVectorNum(ej1,3,7);
    for(loop = 0; loop < 4; loop++)
        printf("sol2 %d \n", solu2[loop]);

    printf(" ------- \n");

    int *solu3;

    solu3 = addZeros(ej1,3,2);
    for(loop = 0; loop < 5; loop++)
        printf("sol3 %d \n", solu3[loop]);

    printf(" ------- \n");
    
    int *solu4;

    solu4 = sumaVectores(ej2,solu3,5,5);
    for(loop = 0; loop < 8; loop++)
        printf("sol4 %d \n", solu4[loop]);
    

    ctimer(&t2,&tucpu,&tscpu);
    printf("Suma = %f \n",alfa);
    printf(" ------- \n");
    printf("Tiempo %f segundos \n",(float) (t2-t1));
    printf(" ------- \n");

    // PRODUCTO MATRIZ // PRODUCTO MATRIZ-VECTOR PARALELO / VECTOR PARALELO //
    /*printf("Empiezo la suma paralela\n");
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
    }*/
    ctimer(&t2,&tucpu,&tscpu);

    // SALIDA DE RESULTADOS //
    printf("Ha terminado la suma paralela\n");
    printf(" ------- \n");
    printf("Suma = %f \n",1);
    printf(" ------- \n");
    // Fin del calculo del Producto Matriz-Vector paralelo
    printf("Tiempo %f segundos \n",(float) (t2-t1));
    printf("He acabado. \n");
    printf("/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/\n");
    printf("/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/\n");
}