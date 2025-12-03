#include "include/Network.hpp"
#include "include/Dataset.hpp"
#include <iostream>
#include <string>
#include <cstdlib>

void printUsage(const char* progName) {
    std::cout << "Uso: " << progName << " [cuda|openmp] [num_models] [num_blocks/teams] [num_threads]\n";
    std::cout << "Exemplo: " << progName << " cuda 1000 32 128\n";
}

int main(int argc, char* argv[]) {
    if (argc < 5) {
        printUsage(argv[0]);
        return 1;
    }

    std::string mode = argv[1];
    int num_models = std::atoi(argv[2]);
    int num_teams = std::atoi(argv[3]);   
    int num_threads = std::atoi(argv[4]);

    std::cout << "Carregando Dataset..." << std::endl;
    // Ajuste o caminho se necessário para "database/iris.txt"
    Neural::Dataset data;
    data.loadInputOutputData(4, 3, "database/iris.txt");
    
    Neural::Network net(data.getInput(), data.getOutput());

    if (mode == "openmp") {
        #ifdef ENABLE_OPENMP
            net.trainOpenMP(num_models, num_teams, num_threads);
        #else
            std::cout << "ERRO: Compilado sem suporte OPENMP (-DENABLE_OPENMP).\n";
        #endif
    } else if (mode == "cuda") {
        #ifdef ENABLE_CUDA
            net.trainCUDA(num_models, num_teams, num_threads);
        #else
            std::cout << "ERRO: Compilado sem suporte CUDA (-DENABLE_CUDA).\n";
        #endif
    } else {
        std::cout << "Modo inválido: use 'cuda' ou 'openmp'.\n";
    }

    return 0;
}