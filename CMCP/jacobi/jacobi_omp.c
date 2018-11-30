#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <omp.h>

static int N;
static int IT_MAXIMAS;
static int SEED;
static int verbose, threads;
static double CONVERGENCIA;

#define SEPARADOR "------------------------------------\n"

// Devuelve el tiempo en segundos
double get_timestamp();

// parsear los argumentos
void parse_arguments(int argc, char *argv[]);

// Ejecución secuencial
// Devuelve el numero de iteraciones
int run(double *A, double *b, double *x_old, double *x_new)
{
  int itr;
  int row, col;
  double dot;
  double dif;
  double sqdif;
  double *xtemp;

  // Bucle hasta convergencia o iteraciones
  itr = 0;
  do
  {
    // Iteracion de jacobi
    for (row = 0; row < N; row++)
    {
      dot = 0.0;
      for (col = 0; col < N; col++)
      {
        if (row != col)
          dot += A[row + col*N] * x_old[col];
      }
      x_new[row] = (b[row] - dot) / A[row + row*N];
    }

    // Intercambio de punteros
    xtemp = x_old;
    x_old      = x_new;
    x_new   = xtemp;

    // convergencia
    sqdif = 0.0;
    for (row = 0; row < N; row++)
    {
      dif    = x_new[row] - x_old[row];
      sqdif += dif * dif;
    }

    itr++;
  } while ((itr < IT_MAXIMAS) && (sqrt(sqdif) > CONVERGENCIA));

  return itr;
}

// Ejecución paralela
// Devuelve el numero de iteraciones
int run_parallel(double *A, double *b, double *x_old, double *x_new, int threads)
{
  int itr;
  int row, col;
  double dot;
  double dif;
  double sqdif;
  double *xtemp;

   // Bucle hasta convergencia o iteraciones
  itr = 0;
  do
  {
    // Iteracion de jacobi
    omp_set_num_threads(threads);
    #pragma omp parallel private (row,col,dot)
    {
      #pragma omp for
      for (row = 0; row < N; row++)
      {
        dot = 0.0;
        for (col = 0; col < N; col++)
        {
          if (row != col)
            dot += A[row + col*N] * x_old[col];
        }
        x_new[row] = (b[row] - dot) / A[row + row*N];
      }
    }

    // Intercambio de punteros
    xtemp = x_old;
    x_old      = x_new;
    x_new   = xtemp;

    // convergencia
    sqdif = 0.0;
    for (row = 0; row < N; row++)
    {
      dif    = x_new[row] - x_old[row];
      sqdif += dif * dif;
    }

    itr++;
  } while ((itr < IT_MAXIMAS) && (sqrt(sqdif) > CONVERGENCIA));

  return itr;
}

int main(int argc, char *argv[])
{

  // parsear los argumentos de entrada
  parse_arguments(argc, argv);

  double *A    = malloc(N*N*sizeof(double));
  double *b    = malloc(N*sizeof(double));
  double *x_old = malloc(N*sizeof(double));
  double *x_new = malloc(N*sizeof(double));

  if(verbose == 1){
    printf(SEPARADOR);
    printf("tamaño de matriz:        %dx%d\n", N, N);
    printf("iteraciones máximas:     %d\n", IT_MAXIMAS);
    printf("límite de convergencia:  %lf\n", CONVERGENCIA);
    printf(SEPARADOR);
  }

  double err;
  double inicio_abs;
  double final_abs;
  double final_sol;
  int itr;
  double inicio_sol;

  //Verbose = 0 -> no hay output
  //Verbose = 1 -> todo el output
  //Verbose = 2 -> output paralelo
  //Verbose = 3 -> output solo secuencial

  if(verbose == 1 || verbose == 3 || verbose == 0){

    inicio_abs = get_timestamp();
    
    // SECUENCIAL

    // Inicializar datos

    srand(SEED);
    for (int row = 0; row < N; row++)
    {
      double rowsum = 0.0;
      for (int col = 0; col < N; col++)
      {
        double value = rand()/(double)RAND_MAX;
        A[row + col*N] = value;
        rowsum += value;
      }
      // Asegurar que es diagonal dominante
      A[row + row*N] += rowsum;
      b[row] = rand()/(double)RAND_MAX;
      x_old[row] = 0.0;
    }
    // ejecutar Jacobi
    inicio_sol = get_timestamp();
    itr = run(A, b, x_old, x_new);
    final_sol = get_timestamp();

    // comprobar error
    err = 0.0;
    for (int row = 0; row < N; row++)
    {
      double tmp = 0.0;
      for (int col = 0; col < N; col++)
      {
        tmp += A[row + col*N] * x_old[col];
      }
      tmp = b[row] - tmp;
      err += tmp*tmp;
    }
    err = sqrt(err);

    final_abs = get_timestamp();
  }

  if(verbose == 1){
    printf("SECUENCIAL\n");
    printf("Error =           %lf\n", err);
    printf("Iteraciones =     %d\n", itr);
    printf("Tiempo total =    %lf seconds\n", (final_abs-inicio_abs));
    printf("Tiempo solución = %lf seconds\n", (final_sol-inicio_sol));
    if (itr == IT_MAXIMAS)
      printf("NO CONVERGENCIA\n");
    printf(SEPARADOR);
  }
  if (verbose == 3) {
    printf("size: \t %d \t iterations: \t %d \t threads: \t %d \t time: \t %lf \t seconds\n", (N*N), itr, threads, (final_sol-inicio_sol));
  }

  if(verbose == 1 || verbose == 2 || verbose == 0){

    // PARALELO

    // Inicializar datos

    srand(SEED);
    for (int row = 0; row < N; row++)
    {
      double rowsum = 0.0;
      for (int col = 0; col < N; col++)
      {
        double value = rand()/(double)RAND_MAX;
        A[row + col*N] = value;
        rowsum += value;
      }
      // Asegurar que es diagonal dominante
      A[row + row*N] += rowsum;
      b[row] = rand()/(double)RAND_MAX;
      x_old[row] = 0.0;
    }
    // Ejecutar jacobi paralelo
    inicio_sol = get_timestamp();
    itr = run_parallel(A, b, x_old, x_new, threads);
    final_sol = get_timestamp();

    // Comprobar error
    err = 0.0;
    for (int row = 0; row < N; row++)
    {
      double tmp = 0.0;
      for (int col = 0; col < N; col++)
      {
        tmp += A[row + col*N] * x_old[col];
      }
      tmp = b[row] - tmp;
      err += tmp*tmp;
    }
    err = sqrt(err);

    final_abs = get_timestamp();
  }

  if(verbose == 1){
    printf("PARALELO\n");
    printf("Error =           %lf\n", err);
    printf("Iteraciones =     %d\n", itr);
    printf("Tiempo total =    %lf seconds\n", (final_abs-inicio_abs));
    printf("Tiempo solucion = %lf seconds\n", (final_sol-inicio_sol));
    if (itr == IT_MAXIMAS)
      printf("NO CONVERGENCIA\n");
    printf(SEPARADOR);
  }
  if (verbose == 2) {
    printf("size: \t %d \t iterations: \t %d \t threads: \t %d \t time: \t %lf \t seconds\n", (N*N), itr, threads, (final_sol-inicio_sol));
  }

  // Liberar memoria

  free(A);
  free(b);
  free(x_old);
  free(x_new);

  return 0;
}


//Métodos entrada

double get_timestamp()
{
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return tv.tv_sec + tv.tv_usec*1e-6;
}

int parse_int(const char *str)
{
  char *next;
  int value = strtoul(str, &next, 10);
  return strlen(next) ? -1 : value;
}

double parse_double(const char *str)
{
  char *next;
  double value = strtod(str, &next);
  return strlen(next) ? -1 : value;
}

void parse_arguments(int argc, char *argv[])
{
  // Valores por defecto
  N = 1000;
  IT_MAXIMAS = 20000;
  CONVERGENCIA = 0.0001;
  SEED = 0;
  verbose = 1;
  threads = 4;

  for (int i = 1; i < argc; i++)
  {
    if (!strcmp(argv[i], "--convergencia") || !strcmp(argv[i], "-c"))
    {
      if (++i >= argc || (CONVERGENCIA = parse_double(argv[i])) < 0)
      {
        printf("Límite de convergencia inválido\n");
        exit(1);
      }
    }
    else if (!strcmp(argv[i], "--iteraciones") || !strcmp(argv[i], "-i"))
    {
      if (++i >= argc || (IT_MAXIMAS = parse_int(argv[i])) < 0)
      {
        printf("Numero de iteraciones inválido\n");
        exit(1);
      }
    }
    else if (!strcmp(argv[i], "--norden") || !strcmp(argv[i], "-n"))
    {
      if (++i >= argc || (N = parse_int(argv[i])) < 0)
      {
        printf("orden de la matriz inválido\n");
        exit(1);
      }
    }
    else if (!strcmp(argv[i], "--seed") || !strcmp(argv[i], "-s"))
    {
      if (++i >= argc || (SEED = parse_int(argv[i])) < 0)
      {
        printf("Seed inválida\n");
        exit(1);
      }
    }
    else if (!strcmp(argv[i], "--verbose") || !strcmp(argv[i], "-v"))
    {
      if (++i >= argc || (verbose = parse_int(argv[i])) < 0)
      {
        printf("Verbose inválido\n");
        exit(1);
      }
    }
    else if (!strcmp(argv[i], "--threads") || !strcmp(argv[i], "-t"))
    {
      if (++i >= argc || (threads = parse_int(argv[i])) < 0)
      {
        printf("threads inválidos\n");
        exit(1);
      }
    }
    else if (!strcmp(argv[i], "--help") || !strcmp(argv[i], "-h"))
    {
      printf("\n");
      printf("Uso: ./jacobi_omp [OPCIONES]\n\n");
      printf("Options:\n");
      printf("  -h  --help               Imprime este mensaje\n");
      printf("  -c  --convergencia C     Elige límite de convergencia\n");
      printf("  -i  --iteraciones  I     Elige numero máximo de iteraciones\n");
      printf("  -n  --norden       N     Elige el orden de la matriz (n)\n");
      printf("  -s  --seed         S     Elige el numero del seed\n");
      printf("  -v  --verbose      V     Elige nivel de verbosidad\n");
      printf("  -t  --threads      T     Elige numero de hilos\n");
      printf("\n");
      exit(0);
    }
    else
    {
      printf("Argumento no reconocido '%s' (prueba '--help')\n", argv[i]);
      exit(1);
    }
  }
}