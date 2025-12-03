# Normaliza as variáveis para aceitar abreviações
VERSION = $(shell echo $(v) | tr '[:upper:]' '[:lower:]')
THREADS = $(t)
PROCESS = $(p)

ifeq ($(VERSION),)
	VERSION = Sequencial
endif
ifeq ($(THREADS),)
	THREADS = $(shell nproc)
endif
# Adiciona um default para PROCESS caso 'p' não seja passado
ifeq ($(PROCESS),)
	PROCESS = 1
endif

# Diretórios
INCLUDE_DIR = include
SRC_DIR_BASE = src

ifeq ($(VERSION),openmp)
	SRC_DIR = $(SRC_DIR_BASE)/srcOpenMP
	COMPILER = g++ -std=c++11 -Wall -g -fopenmp -fcf-protection=none -fno-stack-protector -DNUM_THREADS=$(THREADS)
    # Uncomment the following line if you have offloading configured
    # COMPILER += -foffload=nvptx-none

else ifeq ($(VERSION),cuda)
	SRC_DIR = $(SRC_DIR_BASE)/srcCUDA
	COMPILER = nvcc -std=c++11 -g -arch=sm_60 -D_CUDA -DNUM_THREADS=$(THREADS)
    # Link against cudart is automatic with nvcc

else ifeq ($(VERSION),mpi)
	SRC_DIR = $(SRC_DIR_BASE)/srcMPI
	COMPILER = mpic++ -std=c++11 -Wall -g -fopenmp -D_MPI -DPROCESS=$(PROCESS) -DNUM_THREADS=$(THREADS)

else
	SRC_DIR = $(SRC_DIR_BASE)/srcSequencial
	COMPILER = g++ -std=c++11 -Wall -g -D_SEQUENTIAL -DNUM_THREADS=1
endif

EXEC_PROG = neuralnetwork
BINARIES = $(EXEC_PROG)

# Busca os arquivos fontes e cabeçalhos
SOURCES := $(shell find $(SRC_DIR) -name '*.cpp')
# For CUDA, we might have .cu files
CU_SOURCES := $(shell find $(SRC_DIR) -name '*.cu')

HEADERS := $(shell find $(INCLUDE_DIR) -name '*.hpp')

# Objects
OBJECTS = main.o $(SOURCES:.cpp=.o) $(CU_SOURCES:.cu=.o)

all: clean $(EXEC_PROG)
	@echo Neural Network Build Completed

%.o: %.cpp $(HEADERS)
	$(COMPILER) -I$(INCLUDE_DIR) -c -o $@ $< -w

%.o: %.cu $(HEADERS)
	$(COMPILER) -I$(INCLUDE_DIR) -c -o $@ $< -w

$(EXEC_PROG): $(OBJECTS)
	$(COMPILER) -o $(EXEC_PROG) $(OBJECTS)

# prevents make from getting confused
.PHONY : run
run:
ifeq ($(VERSION),mpi)
	@echo "Executando com $(PROCESS) processo(s) e $(THREADS) thread(s) por processo..."
	mpirun -np $(PROCESS) ./$(EXEC_PROG)
else
	./$(EXEC_PROG)
endif

.PHONY : clean 
clean:
	rm -rf $(EXEC_PROG) $(shell find . -name '*.o')