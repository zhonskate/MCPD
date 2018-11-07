#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <omp.h>

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

	if (carry) printf ("overflow in addition!\n");
}

/* B = n * A */
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
	if (carry) printf ("overflow in multiplication!\n");
}

/* "multiplies" a number by BASEn */
void shift_left (int A[], int n, int N) {
	int	i;

	for (i=N-1; i>=n; i--) A[i] = A[i-n];

	while (i >= 0) A[i--] = 0;
}


/* C = A * B */
void multiply (int A[], int B[], int C[], int N) {
	int	i, j, P[N];

	for (i=0; i<N; i++) {
		/* multiply B by digit A[i] */

		multiply_one_digit (B, P, A[i], N);

		/* shift the partial product left i bytes */

		shift_left (P, i, N);

		/* add result to the running sum */

		add (C, P, C, N);
	}
}


main(int argc, char**argv)
{
    //printf("%s\n", argv[1]);
    int len1 = strlen(argv[1]);
    printf("%d\n",len1);
    //printf("%s\n", argv[2]);
    int len2 = strlen(argv[2]);
    //printf("%d\n",len2);
    int N = len1+len2;
    int A[N], B[N], C[N];

    for(int i=0;i < N; i++){
        A[i] = 0;
        B[i] = 0;
        C[i] = 0;
    }

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
    multiply(A,B,C,N);

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

    // PARALELO

    

    int E[N];
    for(int i=0;i < N; i++)
        E[i] = 0;

    //multiply
    printf("---PARALELO---\n");

    printf("A  [ ");
    for(int loop = N-1; loop >= 0; loop--)
        printf("%d ", A[loop]);
    printf("]\n");

    printf("B  [ ");
    for(int loop = N-1; loop >= 0; loop--)
        printf("%d ", B[loop]);
    //printf("]\nC  [ ");
    printf("]\n");

    omp_set_dynamic(0); 
    omp_set_num_threads(4);

    int D[4*N];
    int n, i, carry,j,sum, P[N], tid, nthreads;
    
    #pragma omp parallel private(i,n, carry, j, sum, P, tid)
    {
        nthreads = omp_get_num_threads();
        for(i=0;i < N*nthreads; i++){
            D[i] = 0;
            E[i] = 0;
        }
        #pragma omp barrier
        tid = omp_get_thread_num();
        //printf("soy el thread %u de %u \n", tid, nthreads);
        for (i=tid; i<N; i=i+nthreads) {
            
            n = A[i];

            //printf("\nthread %d i %d n %d\n",tid,i,n);
            if(tid==0){
                printf("\nthread %d i %d n %d\n",tid,i,n);
                printf("Pbefore  [ ");
                for(int loop = N-1; loop >= 0; loop--)
                    printf("%d ", P[loop]);
                printf("]\n");
            }
            carry = 0;

            for (j=0; j<N; j++) {

                P[j] = n * B[j];
                if (tid==0)
                    printf("\nPJ %d n %d BJ %d\n",P[j],n,B[j]);
                P[j] += carry;

                if (P[j] >= 10) {
                    carry = P[j] / 10;
                    P[j] %= 10;
                } else
                    carry = 0;
            }
            if (carry) printf ("overflow in multiplication!\n");

            /* shift the partial product left i bytes */
            if(tid==0){
                printf("\nthread %d i %d n %d\n",tid,i,n);
                printf("PSH0  [ ");
                for(int loop = N-1; loop >= 0; loop--)
                    printf("%d ", P[loop]);
                printf("]\n");
            }
            for (j=N-1; j>=i; j--) P[j] = P[j-i];

            while (j >= 0) P[j--] = 0;

            /* add result to the running sum */
            if(tid==0){
                printf("\nthread %d i %d n %d\n",tid,i,n);
                printf("P0  [ ");
                for(int loop = N-1; loop >= 0; loop--)
                    printf("%d ", P[loop]);
                printf("]\n");
            }
            //int sum;

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

            if (carry) printf ("overflow in addition!\n");
        }
        #pragma omp barrier
        if(tid==0){
            printf("D  [ ");
            for(int loop = N*nthreads-1; loop >= 0; loop--)
                printf("%d ", D[loop]);
            printf("]\n");

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
                if (carry) printf ("overflow in addition!\n");
            }

            printf("E  [ ");
            for(int loop = N-1; loop >= 0; loop--)
                printf("%d ", E[loop]);
            printf("]\n");

        }

    }
    /*for(int loop = N-1; loop >= 0; loop--)
        printf("%d ", C[loop]);
    printf("]\n");*/

}