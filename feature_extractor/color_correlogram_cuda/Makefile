include ../include/findcudalib.mk

NVCC=nvcc
# internal flags
PREFLAGS:= `pkg-config --cflags opencv` 
POSTFLAGS:=-lcublas `pkg-config --libs opencv`

# Common includes and paths for CUDA
INCLUDES  := -I../include/inc

# CUDA code generation flags
GENCODE_SM10    := -gencode arch=compute_12,code=sm_12
GENCODE_SM20    := -gencode arch=compute_20,code=sm_20
GENCODE_SM30    := -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=\"sm_35,compute_35\"
GENCODE_FLAGS   := $(GENCODE_SM10) $(GENCODE_SM20) $(GENCODE_SM30)

objs=color_correlogram_cuda.o

default:color_correlogram_cuda

all: color_correlogram_cuda

color_correlogram_cuda.o: color_correlogram_cuda.cu
	$(NVCC) $(INCLUDES) $(GENCODE_FLAGS) -c color_correlogram_cuda.cu

color_correlogram_cuda: color_correlogram_cuda.o
	$(NVCC) $(PREFLAGS) -o $(CUDABINSDIR)/color_correlogram_cuda color_correlogram_cuda.o $(POSTFLAGS)
	
clean:
	rm $(objs)
	