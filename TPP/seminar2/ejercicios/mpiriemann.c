#include <mpi.h>
#include <cstdio>
#include <cstdlib>

int main(int argc, char **argv) {
    int master = 0, size, myrank, npoints, npointslocal, i;
    double delta, add, addlocal, x;
    MPI_Init( &argc, &argv );
    MPI_Comm_size( MPI_COMM_WORLD, &size );
    MPI_Comm_rank( MPI_COMM_WORLD, &myrank );
    if (myrank == master ){
        printf("Numbers of divide points:");
        scanf("%ld", &npoints);
    }
    MPI_Bcast( &npoints, 1, MPI_INT, master, MPI_COMM_WORLD);
    delta = 1.0/((double) npoints);
    npointslocal = npoints/size;
    printf(" =================== %ld %ld %ld \n", myrank, npoints, npointslocal);
    addlocal = 0;
    x = myrank * npointslocal * delta;
    for (i = 1; i <= npointslocal; ++i){
        addlocal = addlocal + 1.0/(1+x*x);
        x = x + delta;
    }
    MPI_Reduce( &addlocal, &add, 1, MPI_DOUBLE, MPI_SUM, master, MPI_COMM_WORLD );
    if (myrank == master){
        add = 4.0 * delta * add;
        printf("\nPi = %20.16lf\n", add);
    }   
    MPI_Finalize();
}