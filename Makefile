NVCC ?= nvcc

CUDA_ARCH ?= sm_89
CXXFLAGS ?= -O3 -std=c++17

all: cuda_miner

cuda_miner: cuda_miner.cu
	$(NVCC) $(CXXFLAGS) -arch=$(CUDA_ARCH) cuda_miner.cu -o cuda_miner

clean:
	rm -f cuda_miner
