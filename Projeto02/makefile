# Makefile Híbrido OpenMP/GPU e CUDA

CXX = g++
NVCC = nvcc

# Flags para o compilador C++ (Host)
# -DENABLE_CUDA e -DENABLE_OPENMP habilitam os blocos no main.cpp
# -fopenmp é necessário para o linker do OpenMP
CXX_FLAGS = -O3 -std=c++14 -Iinclude -fopenmp -DENABLE_CUDA -DENABLE_OPENMP

# Flags específicas para OpenMP Offloading (se for usar a versão OpenMP na GPU)
# Nota: No WSL, offloading OpenMP para Nvidia às vezes requer pacotes extras (gcc-offload-nvptx).
# Se der erro de compilação no OpenMP, remova "-foffload=nvptx-none" para testar apenas CUDA.
OMP_OFFLOAD_FLAGS = -foffload=nvptx-none 

# Flags para o compilador CUDA (Device)
# -Xcompiler -fopenmp garante que o nvcc saiba lidar com diretivas omp se houver cabeçalhos mistos
NVCC_FLAGS = -O3 -Iinclude -arch=sm_75 -Xcompiler -fopenmp

# Arquivos
COMMON_SRC = src/Network_Common.cpp src/Dataset.cpp
OMP_SRC = src/Network_OpenMP.cpp
CUDA_SRC = src/Network_CUDA.cu
MAIN_SRC = main.cpp

EXEC = neuralnetwork

all: clean $(EXEC)

$(EXEC):
	@echo "Compilando códigos C++ comuns..."
	$(CXX) $(CXX_FLAGS) -c $(COMMON_SRC)
	
	@echo "Compilando código OpenMP..."
	$(CXX) $(CXX_FLAGS) $(OMP_OFFLOAD_FLAGS) -c $(OMP_SRC)
	
	@echo "Compilando Main..."
	$(CXX) $(CXX_FLAGS) -c $(MAIN_SRC)
	
	@echo "Compilando Kernel CUDA..."
	$(NVCC) $(NVCC_FLAGS) -c $(CUDA_SRC)
	
	@echo "Linkando executável final..."
	# Usamos o nvcc para linkar porque ele resolve automaticamente as bibliotecas do CUDA (cudart)
	$(NVCC) $(NVCC_FLAGS) -o $(EXEC) *.o -lgomp

clean:
	rm -f *.o $(EXEC)