CC=g++
PREFLAGS=`pkg-config --cflags opencv`
POSTFLAGS=`pkg-config --libs opencv`

ROOT=${shell pwd}
export BINSDIR=$(ROOT)/bins
export CUDABINSDIR=$(ROOT)/cubins
export CUDA_PATH=/usr/local/cuda-5.5

CPUDIRS:=$(ROOT)/color_histogram \
	$(ROOT)/color_correlogram \
	$(ROOT)/color_coherence_vector \
	$(ROOT)/color_moment \
	$(ROOT)/texture_lbp \
	$(ROOT)/texture_gabor \
	$(ROOT)/shape_hog \
	$(ROOT)/shape_eoh

CUDADIRS:=$(ROOT)/color_histogram_cuda \
	$(ROOT)/color_correlogram_cuda \
	#$(ROOT)/color_coherence_vector_cuda \
	$(ROOT)/color_moment_cuda \
	$(ROOT)/texture_lbp_cuda \
	#$(ROOT)/texture_gabor_cuda \
	#$(ROOT)/shape_hog_cuda \
	#$(ROOT)/shape_eoh_cuda

default:dir cpu

all:dir cpu cuda

dir:
	mkdir -p $(BINSDIR)
	mkdir -p $(CUDABINSDIR)

cpu:
	@for n in $(CPUDIRS); do $(MAKE) -C $$n; done

cuda:
	@for n in $(CUDADIRS); do $(MAKE) -C $$n; done

clean:
	@for n in $(CPUDIRS); do $(MAKE) -C $$n clean; done
	@for n in $(CUDADIRS); do $(MAKE) -C $$n clean; done
	rm $(BINSDIR)/* $(CUDABINSDIR)/*
