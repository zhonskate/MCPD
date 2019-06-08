CFLAGS          =
FFLAGS          =
CPPFLAGS        =
FPPFLAGS        =
LOCDIR          = ~
MANSEC          = EPS


PETSC_DIR=/opt/petsc-3.8.3
SLEPC_DIR=/opt/slepc-3.8.2

PETSC_ARCH=arch-linux-gnu-c-opt

include ${SLEPC_DIR}/lib/slepc/conf/slepc_common

svd: svd.o chkopts
	-${CLINKER} -o svd svd.o ${SLEPC_SVD_LIB}
	${RM} svd.o

lsqr: lsqr.o chkopts
	-${CLINKER} -o lsqr lsqr.o ${PETSC_KSP_LIB}
	${RM} lsqr.o
