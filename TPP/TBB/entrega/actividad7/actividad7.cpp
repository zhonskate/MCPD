
#include <stdio.h>
#include <stdlib.h>
#include "ctimer.h"
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <cmath>
using namespace tbb;

class MediaDesv{
	int *tam;
	double **M;
	double *media;
	double *desvt;
	public:
	MediaDesv(int *tam, double **M, double *media, double *desvt) :tam(tam), M(M), media(media), desvt(desvt){}
	void operator()(const blocked_range<int> &r) const{
		for(int v = r.begin(); v!=r.end();v++){
			double suma = 0.0;
			for(int i = 0; i < tam[v]; i++){
				suma += M[v][i];
			}
      media[v] = suma /tam[v];
      suma = 0.0;
      for(int i = 0; i < tam[v]; i++){
        suma+=((M[v][i] - media[v]) * (M[v][i] - media[v]));
      }
      desvt[v] = sqrt(suma/tam[v]);
		}
	}
};

int main( int argc, char *argv[] ) {

  double suma;

  if( argc<2 ) {
    printf("Usage: %s n_vectores [tam_max] \n",argv[0]);
    return 1;
  }
  int n_vectores; 
  sscanf(argv[1],"%d",&n_vectores);
  int tam_max = 10000;
  if( argc>2 ) {
    sscanf(argv[2],"%d",&tam_max);
  }

  double **M = (double **) malloc (n_vectores*sizeof(double*));
  int *tam = (int *) malloc (n_vectores*sizeof(int));
  for( size_t v=0; v<n_vectores; v++ ) {
    int t = rand()%tam_max; 
    tam[v] = t>0 ? t : 1;
    M[v] = (double *) malloc (tam[v]*sizeof(double));
    for( size_t i = 0; i<tam[v]; i++ ) {
      M[v][i] = (double) rand()/RAND_MAX * 2.0*tam[v] - 1.0*tam[v];
    }
  }

  double *media_seq = (double *) malloc (n_vectores*sizeof(double));
  double *desvt_seq = (double *) malloc (n_vectores*sizeof(double));
  for( size_t v = 0; v<n_vectores; v++ ) {
    suma = 0.0;
    for( size_t i = 0; i<tam[v]; i++ ) {
      suma += M[v][i];
    }
    media_seq[v] = suma / tam[v];
    suma = 0.0;
    for( size_t i = 0; i<tam[v]; i++ ) {
      suma += ((M[v][i]-media_seq[v])*(M[v][i]-media_seq[v]));
    }
    desvt_seq[v] = sqrt( suma / tam[v] );
  }
  double elapsed, ucpu, scpu;
  double *media = (double *) malloc (n_vectores*sizeof(double));
  double *desvt = (double *) malloc (n_vectores*sizeof(double));
  ctimer(&elapsed,&ucpu,&scpu);

  /****************************************/
  /* CODIGO PARALELO                      */
  /* Calcular la media en el vector media */
  /* y la desviación típica en desvt      */
  /****************************************/

  parallel_for(blocked_range<int>(0,n_vectores),[&](const blocked_range<int> &r ){
    for (int v = r.begin(); v!= r.end(); v++){
      double suma = 0.0;
      for (int i = 0; i < tam[v]; i++){
        suma = suma + M[v][i];
      }
      media[v] = suma / tam[v];
      suma = 0.0;
      for (int i = 0; i < tam[v]; i++){
        suma = suma + pow((M[v][i]-media[v]),2);
      }
      desvt[v] = sqrt(suma/tam[v]);
    }
  });

  //parallel_for(blocked_range<int>(0,n_vectores), MediaDesv(tam,M,media,desvt));


  ctimer(&elapsed,&ucpu,&scpu);
  printf("Tiempo = %f segundos\n",elapsed);

  double error = 0;
  for( size_t i=0; i<n_vectores; i++ ) {
    error += fabs(media_seq[i]-media[i]);
    error += fabs(desvt_seq[i]-desvt[i]);
  }
  printf("Error = %f \n",error);

  free(desvt);
  free(media);
  free(desvt_seq);
  free(media_seq);
  free(tam);
  for( size_t i = 0; i<n_vectores; i++ ) {
    free(M[i]);
  }
  free(M);

  return 0;
}

