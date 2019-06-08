#include <slepcsvd.h>
#include <time.h>
#include <petscksp.h>
#include <petscsys.h>
#include <petscvec.h>
#include <petscmath.h>
#include <slepcsys.h>
#include <slepceps.h>

static char help[] = "Programa para la resolución de imágenes TC con SVD.\n\n";

int main(int argc,char **argv)
{
  Mat            A;               /* operator matrix */
  KSP            ksp;
  PC              pc;
  char filename[PETSC_MAX_PATH_LEN], filename2[PETSC_MAX_PATH_LEN];
  PetscViewer    viewer;
  PetscBool      flg;
  PetscErrorCode ierr;             /* singular value problem solver context */
  //PetscReal      sigma;
  SVDType        type;
  //PetscReal      tol;
  PetscInt       its;
  Vec 	 	 g;

  //ierr = SlepcInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr = PetscPrintf(PETSC_COMM_WORLD,"\nSingular value problem stored in file.\n\n");CHKERRQ(ierr);
  ierr = PetscOptionsGetString(NULL,NULL,"-file",filename,PETSC_MAX_PATH_LEN,&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(PETSC_COMM_WORLD,1,"Must indicate a file name with the -file option");
  ierr = PetscOptionsGetString(NULL,NULL,"-file2",filename2,PETSC_MAX_PATH_LEN,&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(PETSC_COMM_WORLD,1,"Must indicate a file name with the -file2 option");


#if defined(PETSC_USE_COMPLEX)
  ierr = PetscPrintf(PETSC_COMM_WORLD," Reading COMPLEX matrix from a binary file...\n");CHKERRQ(ierr);
#else
  ierr = PetscPrintf(PETSC_COMM_WORLD," Reading REAL matrix from a binary file...\n");CHKERRQ(ierr);
#endif
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,filename,FILE_MODE_READ,&viewer);CHKERRQ(ierr);
  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = MatSetFromOptions(A);CHKERRQ(ierr);
  ierr = MatSetType(A, MATMPIAIJ);
  ierr = MatLoad(A,viewer);CHKERRQ(ierr);

  PetscInt m,n;
  ierr = MatGetSize(A,&m,&n);

  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,filename2,FILE_MODE_READ,&viewer);CHKERRQ(ierr);
  ierr = VecCreate(PETSC_COMM_WORLD, &g); CHKERRQ(ierr);
  ierr = VecSetSizes(g, PETSC_DECIDE,m);
  ierr = VecSetFromOptions(g); CHKERRQ(ierr);
  ierr = VecSetType(g, VECMPI);
  ierr = VecLoad(g, viewer); CHKERRQ(ierr);

  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);


  ierr = KSPCreate(PETSC_COMM_WORLD,&ksp);CHKERRQ(ierr);
  ierr = KSPSetOperators(ksp,A,NULL);CHKERRQ(ierr);
  ierr = KSPSetType(ksp,KSPLSQR);
  ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);



  ierr = KSPGetPC(ksp,&pc);
  ierr = PCSetType(pc, PCNONE); CHKERRQ(ierr);


  //Crear vector solución
  Vec im;
  ierr = VecCreate(PETSC_COMM_WORLD, &im); CHKERRQ(ierr);
  //ierr = VecSetType(im, VECMPI);
  ierr = VecSetFromOptions(im); CHKERRQ(ierr);
  ierr = VecSetSizes(im, PETSC_DECIDE,n);



  //Resolver SVD
  ierr = KSPSolve(ksp, g, im);CHKERRQ(ierr);

  //Extraer estadísticas
  ierr = KSPGetIterationNumber(ksp,&its);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD," Number of iterations of the method: %D\n",its);CHKERRQ(ierr);
  //PetscReal error;
  //ierr = SVDComputeError(svd, i, SVD_ERROR_RELATIVE, &error);
  //ierr = PetscPrintf(PETSC_COMM_WORLD," Error of the method: %D\n",error);CHKERRQ(ierr);
  ierr = KSPGetType(ksp,&type);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD," Solution method: %s\n\n",type);CHKERRQ(ierr);
  //ierr = SVDGetDimensions(svd,&nsv,NULL,NULL);CHKERRQ(ierr);
  //ierr = PetscPrintf(PETSC_COMM_WORLD," Number of requested singular values: %D\n",nsv);CHKERRQ(ierr);
  //ierr = SVDGetTolerances(svd,&tol,&maxit);CHKERRQ(ierr);
  //ierr = PetscPrintf(PETSC_COMM_WORLD," Stopping condition: tol=%.4g, maxit=%D\n",(double)tol,maxit);CHKERRQ(ierr);
  
  //declaro los vectores y les asocio el tamaño correspondiente
  //Vec u,v;
  //ierr = MatCreateVecs(A,&v,&u);

  //Obtengo el resultado im = V*3'*U*g
  /*
  PetscScalar val;
  PetscInt nvd;
  ierr = SVDGetConverged(svd, &nvd);
  for (i=0; i<nvd; i++) {
    ierr = SVDGetSingularTriplet(svd, i, &sigma, u, v);
    ierr = VecDot(u,g,&val);
    sigma = val/sigma;
    ierr = VecAXPY(im,sigma,v); 
  }
  */


  //almaceno la solución
  const char sol[] = "solucion_lsqr.m";
  PetscViewer lab;
  ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD, sol, &lab);
  ierr = PetscViewerPushFormat(lab, PETSC_VIEWER_ASCII_MATLAB);
  ierr = VecView(im,lab);
  ierr = PetscViewerPopFormat(lab);
  
  ierr = PetscViewerDestroy(&lab);CHKERRQ(ierr);
  ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = VecDestroy(&g);CHKERRQ(ierr);
 // ierr = SlepcFinalize();
  ierr = PetscFinalize();
  return ierr;
}
