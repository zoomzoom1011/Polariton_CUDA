CPPFILES = polaron.cpp
CUFILES = polaron_function.cu
OBJECTS = $(CPPFILES:.cpp=.o) $(CUFILES:.cu=.o)
EXECNAME = polaron.exe

CUDADIR = /usr/local/cuda
NVCC = nvcc
MPICC = mpiicpc

INC     = -I$(CUDADIR)/include -I$(MKLROOT)/include -I$(MAGMADIR)/include
FLAGS	= -DMKL_ILP64 -DADD_
LIBS    = -lmkl_gf_ilp64 -lmkl_gnu_thread -lmkl_core -lgomp -lpthread -ldl -lcufft -lmagma \
		-lcublas -lcusparse -lcudart -lmpi
LIBDIRS = -L/${HOME}/intel/mkl/lib/intel64 -L/usr/local/cuda/lib64 -L/${HOME}/magma/lib
INCDIRS = -I/${HOME}/intel/mkl/include -I/usr/local/cuda/include -I${HOME}/magma/include 

${EXECNAME} : ${OBJECTS}
	$(MPICC) -o $(EXECNAME) $(FLAGS) $(LIBDIRS) $(LIBS) $(INCDIRS) $(OBJECTS)
${CPPFILES:.cpp=.o}: ${CPPFILES}
	$(MPICC) -c $(CPPFILES) $(FLAGS) $(LIBDIRS) $(LIBS) $(INCDIRS)
${CUFILES:.cu=.o}: ${CUFILES}
	$(NVCC) -c $(CUFILES) $(FLAGS) $(LIBDIRS) $(LIBS) $(INCDIRS)

clean:
	@echo cleaning up
	@-rm ${OBJECTS}
	@-rm ${EXECNAME}
	@clear