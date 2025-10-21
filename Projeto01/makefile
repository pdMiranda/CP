# Compila a versão sequencial por padrão
VERSION = Sequencial
COMPILER = mpic++ -std=c++11 -Wall -g

# Diretórios
INCLUDE_DIR = include
SRC_DIR_BASE = src
ifeq ($(VERSION),OpenMP)
	SRC_DIR = $(SRC_DIR_BASE)/srcOpenMP
else ifeq ($(VERSION),MPI)
	SRC_DIR = $(SRC_DIR_BASE)/srcMPI
else
	SRC_DIR = $(SRC_DIR_BASE)/srcSequencial
endif

EXEC_PROG = neuralnetwork
BINARIES = $(EXEC_PROG)

# Busca os arquivos fontes e cabeçalhos
SOURCES := $(shell find $(SRC_DIR) -name '*.cpp')
HEADERS := $(shell find $(INCLUDE_DIR) -name '*.hpp')
OBJECTS = main.o $(SOURCES:.cpp=.o)

all: clean $(EXEC_PROG)
	@echo Neural Network Build Completed

%.o: %.cpp $(HEADERS)
	$(COMPILER) -I$(INCLUDE_DIR) -c -o $@ $< -w

$(EXEC_PROG): $(OBJECTS)
	$(COMPILER) -o $(EXEC_PROG) $(OBJECTS)

# prevents make from getting confused
.PHONY : run
run:
	./$(EXEC_PROG)

.PHONY : clean 
clean:
	rm -rf $(EXEC_PROG) $(shell find . -name '*.o')