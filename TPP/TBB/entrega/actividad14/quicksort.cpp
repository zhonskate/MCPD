#include <stdio.h>
#include <stdlib.h>
#include <tbb/parallel_invoke.h>
using namespace tbb;

void quicksort(int *array, int start, int end);
int divide(int *array, int start, int end);

int main() 
{
    const int n = 9;
    const int p = 4;
    int a[] = { 7, 12, 1, -2, 0, 15, 4, 11, 9};

    int i;
    
    printf("\n\nVector desordenado:  ");
    for(i = 0; i < n; ++i)
      printf(" %d ", a[i]);
    printf("\n");
        
    quicksort( a, 0, n-1);

    printf("\n\nVector ordenado:  ");
    for(i = 0; i < n; ++i)
      printf(" %d ", a[i]);
    printf("\n");
    return 0;
}



// Función para dividir el array y hacer los intercambios
int divide(int *array, int start, int end) {
    int left;
    int right;
    int pivot;
    int temp;
 
    pivot = array[start];
    left = start;
    right = end;
 
    // Mientras no se cruzen los índices
    while (left < right) {
        while (array[right] > pivot) {
            right--;
        }
 
        while ((left < right) && (array[left] <= pivot)) {
            left++;
        }
 
        // Si todavía no se cruzan los indices seguimos intercambiando
        if (left < right) {
            temp = array[left];
            array[left] = array[right];
            array[right] = temp;
        }
    }
 
    // Los índices ya se han cruzado, ponemos el pivot en el lugar que le corresponde
    temp = array[right];
    array[right] = array[start];
    array[start] = temp;
 
    // La nueva posición del pivot
    return right;
}
 
// Función recursiva para hacer el ordenamiento
void quicksort(int *array, int start, int end)
{
    int pivot;
 
    if (start < end) {
        pivot = divide(array, start, end);
 
        // Ordeno la lista de los mayores
        // Ordeno la lista de los menores y mayores
        parallel_invoke([&]{quicksort(array, start, pivot - 1);},[&]{quicksort(array, pivot + 1, end);});
    }
}



