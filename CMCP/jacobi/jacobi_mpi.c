#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <mpi.h>
#include <sys/time.h>

double Convergencia(double *x_old, double *x_new, int N);

double get_timestamp();

main(int argc, char** argv) {

  //Inicialización de variables
  MPI_Status status;     
  int N=atoi(argv[3]), NoofRows_Bloc, NoofRows=atoi(argv[3]), NoofCols=atoi(argv[3]);
  int Numprocs, MyRank, Root=0, verbose=atoi(argv[2]);
  int row, col, index, Iteration, GlobalRowNo;

  double *A_inicial, *A, *b, *ARecv, *BRecv;
  double *x_new, *x_old, *Bloc_X, tmp;

  int IT_MAXIMAS = atoi(argv[1]);

  // Inicialización MPI
  MPI_Init(&argc, &argv); 
  MPI_Comm_rank(MPI_COMM_WORLD, &MyRank);
  MPI_Comm_size(MPI_COMM_WORLD, &Numprocs);
  
  // Inicialización datos
  if(MyRank == Root) {

     b  = (double *)malloc(N*sizeof(double));
     A_inicial  = (double *)malloc(N*N*sizeof(double));
    
    index = 0;
    for (row = 0; row < N; row++)
    {
      double rowsum = 0.0;
      for (col = 0; col < N; col++)
      {
        double value = rand()/(double)RAND_MAX;
        A_inicial[row + col*N] = value;
        rowsum += value;
      }
      A_inicial[row + row*N] += rowsum;
      b[row] = rand()/(double)RAND_MAX;
    }

	 // Convertir A_inicial en un vector para distribuirlo mas facilmente 
    A = (double *)malloc(N*N*sizeof(double));
	  index    = 0;
	  for(row=0; row<N; row++)
	  	  for(col=0; col<N; col++)
			  A[index++] = A_inicial[row + col*N];
    

  }

  double inicio_sol = get_timestamp();

  MPI_Bcast(&NoofRows, 1, MPI_INT, Root, MPI_COMM_WORLD); 
  MPI_Bcast(&NoofCols, 1, MPI_INT, Root, MPI_COMM_WORLD); 

  if(NoofRows != NoofCols) {
	  MPI_Finalize();
	  if(MyRank == 0){
	  	  printf("LA matriz debe ser cuadrada... \n");
	  }
	  exit(-1);
  }  	

  // Enviar N (tamaño de matriz)
  MPI_Bcast(&N, 1, MPI_INT, Root, MPI_COMM_WORLD); 

  if(N % Numprocs != 0) {
	  MPI_Finalize();
	  if(MyRank == 0){
	  	  printf("La matriz no se puede distribuir regularmente \n");
	  }
	  exit(-1);
  }  	

  NoofRows_Bloc = N/Numprocs;
  //Tamaño de los buffers para recibir los datos 
  ARecv = (double *) malloc (NoofRows_Bloc * N* sizeof(double));
  BRecv = (double *) malloc (NoofRows_Bloc * sizeof(double));

  // Enviar A y b
  MPI_Scatter (A, NoofRows_Bloc * N, MPI_DOUBLE, ARecv, NoofRows_Bloc * N, 
					MPI_DOUBLE, 0, MPI_COMM_WORLD);

  MPI_Scatter (b, NoofRows_Bloc, MPI_DOUBLE, BRecv, NoofRows_Bloc, MPI_DOUBLE, 0, 
					MPI_COMM_WORLD);

  x_new  = (double *) malloc (N * sizeof(double));
  x_old  = (double *) malloc (N * sizeof(double));
  Bloc_X = (double *) malloc (NoofRows_Bloc * sizeof(double));

  // Inicializar x a 0
  for(row=0; row<NoofRows; row++)
	  x_new[row] = 0.0;

  Iteration = 0;
  do{
      //swap
	   for(row=0; row<N; row++)
			 x_old[row] = x_new[row];

      //Cálculo Jacobi
      for(row=0; row<NoofRows_Bloc; row++){

          GlobalRowNo = (MyRank * NoofRows_Bloc) + row;
			 Bloc_X[row] = BRecv[row];
			 index = row * N;

			 for(col=0; col<GlobalRowNo; col++){
				 Bloc_X[row] -= x_old[col] * ARecv[index + col];
			 }
			 for(col=GlobalRowNo+1; col<N; col++){
				 Bloc_X[row] -= x_old[col] * ARecv[index + col];
			 }
          Bloc_X[row] = Bloc_X[row] / ARecv[row*N + GlobalRowNo];
		}

      //envío de datos
  		MPI_Allgather(Bloc_X, NoofRows_Bloc, MPI_DOUBLE, x_new, 
						  NoofRows_Bloc, MPI_DOUBLE, MPI_COMM_WORLD);
      Iteration++;
  }while( (Iteration < IT_MAXIMAS)); 
  //&& (Convergencia(x_old, x_new, N) >= 1.0E-24)
  
  double final_sol = get_timestamp();

  //Imprimir resultados

  //Verbose = 0 -> no hay output
  //Verbose = 1 -> todo el output
  //Verbose = 2 -> output paralelo

  if (MyRank == 0 && verbose == 1) {

     printf ("\n");
     printf(" ------------------------------------------- \n");
     printf("Resultados de Jacobi en el hilo %d: \n", MyRank);
     printf ("\n");

     printf("Matriz A\n");
     printf ("\n");
     for (row = 0; row < N; row++) {
        for (col = 0; col < N; col++)
	        printf("%.3lf  ", A_inicial[row + col*N]);
        printf("\n");
     }
     printf ("\n");
     printf("Matriz b \n");
     printf("\n");
     for (row = 0; row < N; row++) {
         printf("%.3lf\n", b[row]);
     }
     printf ("\n");
     printf("Vector solución \n");
	  printf("Number of iterations = %d\n",Iteration);
     printf ("\n");
     for(row = 0; row < N; row++)
        printf("%.12lf\n",x_new[row]);
     printf(" --------------------------------------------------- \n\n");
     printf("Tiempo solución = %lf seconds\n\n", (final_sol-inicio_sol));
     printf(" --------------------------------------------------- \n");
  }

  if (MyRank == 0 && verbose == 2) {
    printf("size: \t %d \t iterations: \t %d \t threads \t %d \t time: \t %lf \t seconds\n", (N*N), Iteration, Numprocs, (final_sol-inicio_sol));
  }
  MPI_Finalize(); 
}

//comprobar convergencia
double Convergencia(double *x_old, double *x_new, int N)
{
   int  index;
	double Sum;

   Sum = 0.0;
	for(index=0; index<N; index++)
		 Sum += (x_new[index] - x_old[index])*(x_new[index]-x_old[index]);

   return(Sum);
}

double get_timestamp()
{
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return tv.tv_sec + tv.tv_usec*1e-6;
}


