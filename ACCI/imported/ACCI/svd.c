#include <slepcsvd.h>

static char help[] = "Solves a singular value problem with the matrix loaded from a file.\n"
  "This example works for both real and complex numbers.\n\n"
  "The command line options are:\n"
  "  -file <filename>, where <filename> = matrix file in PETSc binary form.\n\n";

int main(int argc,char **argv)
{
  Mat            A;               /* operator matrix */
  SVD            svd;             /* singular value problem solver context */
  SVDType        type;
  PetscReal      tol;
  PetscInt       nsv,maxit,its,nconv,i;
  char           filenameA[PETSC_MAX_PATH_LEN];
  char           filenameG[PETSC_MAX_PATH_LEN];
  PetscViewer    viewer;
  PetscBool      flg,terse;
  PetscErrorCode ierr;
  PetscScalar    aux;
  PetscReal      error,sigma,mu=PETSC_SQRT_MACHINE_EPSILON;

  Vec            u,v,g,im;

  ierr = SlepcInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        Load the operator matrix that defines the singular value problem
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = PetscPrintf(PETSC_COMM_WORLD,"\nSingular value problem stored in file.\n\n");CHKERRQ(ierr);
  ierr = PetscOptionsGetString(NULL,NULL,"-fileA",filenameA,PETSC_MAX_PATH_LEN,&flg);CHKERRQ(ierr);
  ierr = PetscOptionsGetString(NULL,NULL,"-fileG",filenameG,PETSC_MAX_PATH_LEN,&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(PETSC_COMM_WORLD,1,"Must indicate a file name with the -file option");

#if defined(PETSC_USE_COMPLEX)
  ierr = PetscPrintf(PETSC_COMM_WORLD," Reading COMPLEX matrix from a binary file...\n");CHKERRQ(ierr);
#else
  ierr = PetscPrintf(PETSC_COMM_WORLD," Reading REAL matrix from a binary file...\n");CHKERRQ(ierr);
#endif

  ierr = PetscPrintf(PETSC_COMM_WORLD," ********************  1  ********************\n");CHKERRQ(ierr);
  // Load A
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,filenameA,FILE_MODE_READ,&viewer);CHKERRQ(ierr);
  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = MatSetFromOptions(A);CHKERRQ(ierr);
  ierr = MatLoad(A,viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);

  ierr = PetscPrintf(PETSC_COMM_WORLD," ********************  2  ********************\n");CHKERRQ(ierr);

  // Load b
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,filenameG,FILE_MODE_READ,&viewer);CHKERRQ(ierr);
  ierr = VecCreate(PETSC_COMM_WORLD,&g);CHKERRQ(ierr);
  ierr = VecSetFromOptions(g);CHKERRQ(ierr);
  ierr = VecLoad(g,viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);

  ierr = PetscPrintf(PETSC_COMM_WORLD," ********************  3  ********************\n");CHKERRQ(ierr);

  // initialize v, u
  ierr = MatCreateVecs(A,&v,&u);CHKERRQ(ierr);

  ierr = PetscPrintf(PETSC_COMM_WORLD," ********************  3.1  ********************\n");CHKERRQ(ierr);

  ierr = VecCreate(PETSC_COMM_WORLD,&im);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD," ********************  3.2  ********************\n");CHKERRQ(ierr);
  ierr = VecSetSizes(im,PETSC_DECIDE,16384);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD," ********************  3.3  ********************\n");CHKERRQ(ierr);
  ierr = VecSetType(im,VECMPI);CHKERRQ(ierr);
  ierr = VecSet(im,0.0);CHKERRQ(ierr);


  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                Create the singular value solver and set various options
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /*
     Create singular value solver context
  */

  ierr = PetscPrintf(PETSC_COMM_WORLD," ********************  4  ********************\n");CHKERRQ(ierr);

  ierr = SVDCreate(PETSC_COMM_WORLD,&svd);CHKERRQ(ierr);

  /*
     Set operator
  */
  ierr = PetscPrintf(PETSC_COMM_WORLD," ********************  5  ********************\n");CHKERRQ(ierr);
   
  ierr = SVDSetOperator(svd,A);CHKERRQ(ierr);

  /*
     Set solver parameters at runtime
  */
  ierr = PetscPrintf(PETSC_COMM_WORLD," ********************  6  ********************\n");CHKERRQ(ierr);
  ierr = SVDSetFromOptions(svd);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                      Solve the singular value system
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = SVDSolve(svd);CHKERRQ(ierr);
  ierr = SVDGetIterationNumber(svd,&its);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD," Number of iterations of the method: %D\n",its);CHKERRQ(ierr);

  /*
     Optional: Get some information from the solver and display it
  */
  ierr = SVDGetType(svd,&type);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD," Solution method: %s\n\n",type);CHKERRQ(ierr);
  ierr = SVDGetDimensions(svd,&nsv,NULL,NULL);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD," Number of requested singular values: %D\n",nsv);CHKERRQ(ierr);
  ierr = SVDGetTolerances(svd,&tol,&maxit);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD," Stopping condition: tol=%.4g, maxit=%D\n",(double)tol,maxit);CHKERRQ(ierr);

 /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                    Display solution and clean up
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /*
     Get number of converged singular triplets
  */
  ierr = SVDGetConverged(svd,&nconv);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD," Number of converged approximate singular triplets: %D\n\n",nconv);CHKERRQ(ierr);

  if (nconv>0) {
    /*
       Display singular values and relative errors
    */
    ierr = PetscPrintf(PETSC_COMM_WORLD,
         "          sigma           relative error\n"
         "  --------------------- ------------------\n");CHKERRQ(ierr);
    for (i=0;i<nconv;i++) {
      /*
         Get converged singular triplets: i-th singular value is stored in sigma
      */
      ierr = SVDGetSingularTriplet(svd,i,&sigma,u,v);CHKERRQ(ierr);

      ierr = VecDot(u,g,&aux);CHKERRQ(ierr);

      aux = aux/sigma;

      ierr = VecAXPY(im,aux,v);CHKERRQ(ierr);

      /*
         Compute the error associated to each singular triplet
      */
      ierr = SVDComputeError(svd,i,SVD_ERROR_RELATIVE,&error);CHKERRQ(ierr);

      ierr = PetscPrintf(PETSC_COMM_WORLD,"       % 6f      ",(double)sigma);CHKERRQ(ierr);
      ierr = PetscPrintf(PETSC_COMM_WORLD," % 12g\n",(double)error);CHKERRQ(ierr);
    }
    ierr = PetscPrintf(PETSC_COMM_WORLD,"\n");CHKERRQ(ierr);
    PetscViewerASCIIOpen(PETSC_COMM_WORLD, "sol.m", &viewer);
    PetscViewerPushFormat(viewer, PETSC_VIEWER_ASCII_MATLAB);
    VecView(im,viewer);
    PetscViewerPopFormat(viewer);
    PetscViewerDestroy(&viewer);
  }

  /*
     Free work space
  */
  ierr = SVDDestroy(&svd);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = VecDestroy(&u);CHKERRQ(ierr);
  ierr = VecDestroy(&v);CHKERRQ(ierr);
  ierr = SlepcFinalize();
  return ierr;
}
