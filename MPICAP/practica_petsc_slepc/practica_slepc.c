#include <time.h>
#include <petscksp.h>
#include <petscsys.h>
#include <petscvec.h>
#include <petscmath.h>
#include <slepcsys.h>
#include <slepceps.h>

static char help[] = "Programa para resolver el problema de valores propios asociado a reactor nuclear.\n\n";


typedef struct{
   /* Completar: Definir la estructura de la matriz con las matrices y vectores internos*/
  Mat L11;
  Mat L22;
  Vec L21;
  Vec M11;
  Vec M12;
  Vec Vt;

}MatA;



int MatMult_ringhals(Mat A, Vec x, Vec y){

  /*Completar: Definir la multiplicaci�n y=Ax para la matriz shell A*/
  PetscErrorCode ierr;
  KSP                solver,solver2;

  MatA *ctx;
  void *ptr;
  
  
  MatShellGetContext(A,&ptr);
  ctx = (MatA *)ptr;
  
  /* Crear y configurar los solvers KSP para los sistemas de ecuaciones*/

  ierr = KSPCreate(PETSC_COMM_WORLD,&solver);;CHKERRQ(ierr);

  ierr = KSPSetOperator(solver,ctx->L22,ctx->L22);CHKERRQ(ierr);

  ierr = KSPSetType(solver,KSPCG);CHKERRQ(ierr);

  ierr = KSPCreate(PETSC_COMM_WORLD,&solver2);;CHKERRQ(ierr);

  ierr = KSPSetOperator(solver2,ctx->L11,ctx->L11);CHKERRQ(ierr);

  ierr = KSPSetType(solver2,KSPCG);CHKERRQ(ierr);


  /* w = L21*x */
  ierr = VecPointwiseMult(ctx->Vt, ctx->L21, x);CHKERRQ(ierr);

  /* y = L22^-1 * w */
  ierr = KSPSolve(solver, ctx->Vt, y);CHKERRQ(ierr);
  
  /* w = M12*y */
  ierr = VecPointwiseMult(ctx->Vt, ctx->M12, y);CHKERRQ(ierr);
  
  /* y = M11*x */
  ierr = VecPointwiseMult(y, ctx->M11, x);CHKERRQ(ierr);
  
  /* w = w + y */
  PetscScalar alpha = 1;

  ierr = VecAXPY(ctx->Vt,alpha,x);
  
  /* y = L11^-1 * w */
  ierr = KSPSolve(solver2, ctx->Vt, y);CHKERRQ(ierr);

  return 0;


}


int main(int argc, char **argv){


  PetscErrorCode ierr;
  char           filename[PETSC_MAX_PATH_LEN];
  
  
  Mat            Amat; //Matriz Shell
  MatA           A;  // Matriz del tipo que hemos definido para cargar los datos
  
  PetscViewer     fd;
  PetscInt        n,m;
  
  PetscReal      error,tol,re,im;
  PetscInt       nev,maxit,its,nconv,size;
  
  clock_t ini, fin ;
  
  
  SlepcInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;

  sscanf(argv[1],"%s",filename);
  
  //Viewer para cargar el fichero
	ierr  = PetscViewerBinaryOpen(PETSC_COMM_WORLD,filename,FILE_MODE_READ,&fd);CHKERRQ(ierr);
 
  /* Completar: Cargar Matrices y Vectores por orden del fichero ringhals1.petsc o ringhals2.petsc */

  ierr = MatCreate(PETSC_COMM_WORLD,&A.L11);CHKERRQ(ierr);
  ierr = MatSetFromOptions(A.L11);CHKERRQ(ierr);
  ierr = MatSetType(A.L11,MATMPIAIJ);CHKERRQ(ierr);
  ierr = MatGetSize(A.L11,&m,&n);
  ierr = MatLoad(A.L11,fd);CHKERRQ(ierr);

  ierr = MatCreate(PETSC_COMM_WORLD,&A.L22);CHKERRQ(ierr);
  ierr = MatSetFromOptions(A.L22);CHKERRQ(ierr);
  ierr = MatSetType(A.L22,MATMPIAIJ);CHKERRQ(ierr);
  ierr = MatLoad(A.L22,fd);CHKERRQ(ierr);

  ierr = VecCreate(PETSC_COMM_WORLD,&A.L21);CHKERRQ(ierr);
  ierr = VecSetFromOptions(A.L21);CHKERRQ(ierr);
  ierr = VecLoad(A.L21,fd);CHKERRQ(ierr);

  ierr = VecCreate(PETSC_COMM_WORLD,&A.M11);CHKERRQ(ierr);
  ierr = VecSetFromOptions(A.M11);CHKERRQ(ierr);
  ierr = VecLoad(A.M11,fd);CHKERRQ(ierr);

  ierr = VecCreate(PETSC_COMM_WORLD,&A.M12);CHKERRQ(ierr);
  ierr = VecSetFromOptions(A.M12);CHKERRQ(ierr);
  ierr = VecLoad(A.M12,fd);CHKERRQ(ierr);  

  ierr = MatCreateVecs(A.L11,NULL,&A.Vt);CHKERRQ(ierr);

  // Destroy viewer
	ierr = PetscViewerDestroy(&fd);CHKERRQ(ierr);
 

  ini=clock();
  
  /* Completar: Crear matriz Shell y redefinir producto */

  

  
  /* Completar: Crear contexto solver EPS*/


  /* Completar: Aplicar el solver */

  fin=clock();
  
  /* Obtener la soluc�n y estadistcas */
  ierr = EPSGetIterationNumber(eps,&its);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD," Number of iterations of the method: %D\n",its);CHKERRQ(ierr);

  ierr = EPSGetType(eps,&type);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD," Solution method: %s\n\n",type);CHKERRQ(ierr);
  ierr = EPSGetDimensions(eps,&nev,NULL,NULL);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD," Number of requested eigenvalues: %D\n",nev);CHKERRQ(ierr);
  ierr = EPSGetTolerances(eps,&tol,&maxit);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD," Stopping condition: tol=%.4g, maxit=%D\n",(double)tol,maxit);CHKERRQ(ierr);
  
  PetscPrintf(PETSC_COMM_WORLD, "Total time = %9.3f seconds\n", ((double)(fin-ini)*1000.0/ CLOCKS_PER_SEC)/1000.0);
  
  
  ierr = EPSDestroy(&eps);CHKERRQ(ierr);
  ierr = MatDestroy(&Amat);CHKERRQ(ierr);
  SlepcFinalize();
  return ierr;
}

