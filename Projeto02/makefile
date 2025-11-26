# Valores padrão
NUM_TEAMS ?= 4
NUM_THREADS ?= 32

# Compiladores
NVCC = nvcc
CXX = g++

# Flags de compilação
CXXFLAGS = -Iinclude -O2 -fopenmp -DNUM_TEAMS=$(NUM_TEAMS) -DNUM_THREADS=$(NUM_THREADS)
CXXFLAGS_GPU = -Iinclude -O2 -fopenmp -foffload=nvptx-none -fno-lto -DNUM_TEAMS=$(NUM_TEAMS) -DNUM_THREADS=$(NUM_THREADS)
NVCCFLAGS = -Iinclude -O2 -Xcompiler -fopenmp

# Diretórios
SRC_CUDA = srcCuda
SRC_CPP = srcOpenMP_GPU
INCLUDE = include

# Arquivos de saída
CUDA_OUT = cuda_program
CPP_OUT = cpp_program
CPP_GPU_OUT = cpp_gpu_program

# Regras
all: $(CUDA_OUT) $(CPP_OUT) $(CPP_GPU_OUT)

# Versão CUDA
$(CUDA_OUT): main.cpp $(SRC_CUDA)/Network.cu $(SRC_CPP)/Dataset.cpp
	$(NVCC) $(NVCCFLAGS) -o $@ $^

# Versão OpenMP CPU
$(CPP_OUT): main.cpp $(SRC_CPP)/Network.cpp $(SRC_CPP)/Dataset.cpp
	$(CXX) $(CXXFLAGS) -o $@ $^

# Versão OpenMP GPU
$(CPP_GPU_OUT): main.cpp $(SRC_CPP)/Network.cpp $(SRC_CPP)/Dataset.cpp
	$(CXX) $(CXXFLAGS_GPU) -o $@ $^

clean:
	rm -f $(CUDA_OUT) $(CPP_OUT) $(CPP_GPU_OUT)

.PHONY: all clean