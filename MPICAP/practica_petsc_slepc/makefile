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

practica_slepc: practica_slepc.o chkopts
	-${CLINKER} -o practica_slepc practica_slepc.o ${SLEPC_EPS_LIB}
	${RM} practica_slepc.o
