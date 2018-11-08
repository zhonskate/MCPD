#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#include "ctimer.h"

void add (int A[], int B[], int C[], int N) {
	int	i, carry, sum;

	carry = 0;

	for (i=0; i<N; i++) {

		sum = A[i] + B[i] + carry;

		if (sum >= 10) {
			carry = 1;
			sum -= 10;
		} else
			carry = 0;

		C[i] = sum;
	}
}

void multiply_one_digit (int A[], int B[], int n, int N) {
	int	i, carry;

	carry = 0;

	for (i=0; i<N; i++) {

		B[i] = n * A[i];

		B[i] += carry;

		if (B[i] >= 10) {      
			carry = B[i] / 10;
			B[i] %= 10;
		} else
			carry = 0;
	}
}

void shift_left (int A[], int n, int N) {
	int	i;

	for (i=N-1; i>=n; i--) A[i] = A[i-n];
	while (i >= 0) A[i--] = 0;
}


void multiply (int A[], int B[], int C[], int N) {
	int	i, j, P[N];

	for (i=0; i<N; i++) {
		multiply_one_digit (B, P, A[i], N);
		shift_left (P, i, N);
		add (C, P, C, N);
	}
}


main(int argc, char**argv)
{
    // DECLARACION DE VARIABLES

    double t1,t2,tucpu,tscpu;
    int len1 = strlen(argv[1]);
    int len2 = strlen(argv[2]);
    int N = len1+len2;
    int A[N], B[N], C[N];

    for(int i=0;i < N; i++){
        A[i] = 0;
        B[i] = 0;
        C[i] = 0;
    }

    // RELLENADO DE MATRICES

    char k[len1];
    strcpy(k, argv[1]);
    for(int i=0;i < len1; i++){
        A[i] = k[len1-1-i] - '0';
    }
    char l[len2];
    strcpy(l, argv[2]);
    for(int i=0;i < len2; i++){
        B[i] = l[len2-1-i] - '0';
    }

    // SECUENCIAL
    ctimer(&t1,&tucpu,&tscpu);
    multiply(A,B,C,N);
    ctimer(&t2,&tucpu,&tscpu);

    printf("---SECUENCIAL---\n");

    printf("A  [ ");
    for(int loop = N-1; loop >= 0; loop--)
        printf("%d ", A[loop]);
    printf("]\n");

    printf("B  [ ");
    for(int loop = N-1; loop >= 0; loop--)
        printf("%d ", B[loop]);
    printf("]\nC  [ ");

    for(int loop = N-1; loop >= 0; loop--)
        printf("%d ", C[loop]);
    printf("]\n");

    printf(" ------- \n");
    printf("Tiempo %f segundos \n",(float) (t2-t1));
    printf(" ------- \n");


    // PARALELO

    printf("---PARALELO---\n");

    omp_set_num_threads(4);

    int D[4*N];
    int n, i, carry,j,sum, P[N], tid, nthreads;
    int E[N];
    for(int i=0;i < N; i++)
        E[i] = 0;
    
    ctimer(&t1,&tucpu,&tscpu);

    #pragma omp parallel shared (B,A) private(i,n, carry, j, sum, P, tid)
    {
        nthreads = omp_get_num_threads();
        for(i=0;i < N*nthreads; i++){
            D[i] = 0;
        }
        #pragma omp barrier
        tid = omp_get_thread_num();

        for (i=tid; i<len1; i=i+nthreads) {
            
            n = A[i];

            carry = 0;

            for (j=0; j<N; j++) {

                P[j] = n * B[j];

                P[j] += carry;

                if (P[j] >= 10) {
                    carry = P[j] / 10;
                    P[j] %= 10;
                } else
                    carry = 0;
            }

            // SHIFT

            for (j=N-1; j>=i; j--) P[j] = P[j-i];
            while (j >= 0) P[j--] = 0;

            // SUMA Y ACUMULACION EN D

            carry = 0;
            sum = 0;

            for (j=0; j<N; j++) {

                sum = D[tid*N+j] + P[j] + carry;

                if (sum >= 10) {
                    carry = 1;
                    sum -= 10;
                } else
                    carry = 0;

                D[tid*N+j] = sum;
            }
        }

        #pragma omp barrier

        // TRANSFERENCIA A E SUMANDO PARCIALES

        if(tid==0){

            for(int k=0; k<nthreads;k++){

                carry = 0;
                sum = 0;

                for (j=0; j<N; j++) {

                    sum = E[j] + D[k*N+j] + carry;

                    if (sum >= 10) {
                        carry = 1;
                        sum -= 10;
                    } else
                        carry = 0;

                    E[j] = sum;
                }
            }
        }       
    }

    ctimer(&t2,&tucpu,&tscpu);

    printf("A  [ ");
    for(int loop = N-1; loop >= 0; loop--)
        printf("%d ", A[loop]);
    printf("]\n");

    printf("B  [ ");
    for(int loop = N-1; loop >= 0; loop--)
        printf("%d ", B[loop]);
    printf("]\n");

    printf("C  [ ");
    for(int loop = N-1; loop >= 0; loop--)
        printf("%d ", E[loop]);
    printf("]\n");

    printf(" ------- \n");
    printf("Tiempo %f segundos \n",(float) (t2-t1));
    printf(" ------- \n");
}