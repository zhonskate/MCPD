#include <string.h>
#include <stdlib.h>

void add (int A[], int B[], int C[], int N, int BASE) {
	int	i, carry, sum;

	/* no carry yet */

	carry = 0;

	/* go from least to most significant digit */

	for (i=0; i<N; i++) {

		/* the i'th digit of C is the sum of the
		 * i'th digits of A and B, plus any carry
		 */
		sum = A[i] + B[i] + carry;

		/* if the sum exceeds the base, then we have a carry. */

		if (sum >= BASE) {

			carry = 1;

			/* make sum fit in a digit (same as sum %= BASE) */

			sum -= BASE;
		} else
			/* otherwise no carry */

			carry = 0;

		/* put the result in the sum */

		C[i] = sum;
	}

	/* if we get to the end and still have a carry, we don't have
	 * anywhere to put it, so panic! 
	 */
	if (carry) printf ("overflow in addition!\n");
}

/* B = n * A */
void multiply_one_digit (int A[], int B[], int n, int N, int BASE) {
	int	i, carry;

	/* no extra overflow to add yet */

	carry = 0;

	/* for each digit, starting with least significant... */

	for (i=0; i<N; i++) {

		/* multiply the digit by n, putting the result in B */

		B[i] = n * A[i];

		/* add in any overflow from the last digit */

		B[i] += carry;

		/* if this product is too big to fit in a digit... */

		if (B[i] >= BASE) {

			/* handle the overflow */

			carry = B[i] / BASE;
			B[i] %= BASE;
		} else

			/* no overflow */

			carry = 0;
	}
	if (carry) printf ("overflow in multiplication!\n");
}

/* "multiplies" a number by BASEn */
void shift_left (int A[], int n, int N) {
	int	i;

	/* going from left to right, move everything over to the
	 * left n spaces
	 */
	for (i=N-1; i>=n; i--) A[i] = A[i-n];

	/* fill the last n digits with zeros */

	while (i >= 0) A[i--] = 0;
}


/* C = A * B */
void multiply (int A[], int B[], int C[], int N, int BASE) {
	int	i, j, P[N];

	for (i=0; i<N; i++) {
		/* multiply B by digit A[i] */

		multiply_one_digit (B, P, A[i], N, BASE);

		/* shift the partial product left i bytes */

		shift_left (P, i, N);

		/* add result to the running sum */

		add (C, P, C, N, BASE);
	}
}


main(int argc, char**argv)
{
    //printf("%s\n", argv[1]);
    int len1 = strlen(argv[1]);
    //printf("%d\n",len1);
    //printf("%s\n", argv[2]);
    int len2 = strlen(argv[2]);
    //printf("%d\n",len2);
    int N = len1+len2;
    static int BASE = 10;
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


    printf("A ");
    for(int loop = N-1; loop >= 0; loop--)
        printf("%d ", A[loop]);
    printf("\n");

    printf("B ");
    for(int loop = N-1; loop >= 0; loop--)
        printf("%d ", B[loop]);
    printf("\n");

    multiply(A,B,C,N,BASE);

    printf("C ");
    for(int loop = N-1; loop >= 0; loop--)
        printf("%d ", C[loop]);
    printf("\n");

}