
#include <stdio.h>
#include <iostream>
using namespace std;

#define BUFFER_SIZE 100

int main( )  {

  FILE* input_file;
  FILE* output_file;
  char fichero_entrada[100];
  char fichero_salida[100];

  cout << "Introduce el fichero de entrada: " ;
  cin >> fichero_entrada;
  if( ( input_file = fopen(fichero_entrada,"r") ) == NULL ) {
    cout << "Fichero no existente." << endl;
    return 0;
  }
  cout << "Introduce el fichero de salida: " ;
  cin >> fichero_salida;
  output_file = fopen(fichero_salida,"w");

  /*************************/
  /* Resolucion secuencial */
  char buffer[BUFFER_SIZE];
  size_t n;
  bool primero = false;
  do {
    n = fread( buffer, 1, BUFFER_SIZE, input_file );
    if( primero ) buffer[0] = toupper(buffer[0]);
    primero = false;
    for( int i=1; i<n; i++ ) {
      char *c = &(buffer[i-1]);
      if( *c == ' ' || *c == '\r' || *c == '\n' ) {
        *c = toupper(*(++c));
      }
    }
    fwrite( buffer, 1, n, output_file );
    if( buffer[n-1] == ' ' ) primero = true;
  } while ( n );
  /*************************/
  
  fclose(input_file);
  fclose(output_file);
}



