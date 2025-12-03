#include "include/Network.hpp"
#include "include/Dataset.hpp"
#include <chrono>
#include <omp.h>
#include <iostream>
#include <string>
#include <unistd.h>

#ifdef _MPI
#include <mpi.h>
#endif

using namespace std;

int main(int argc, char* argv[])
{
    int num_threads = 4; // Default
    int num_models = 1;  // Default
    
    // Parse arguments
    int opt;
    while ((opt = getopt(argc, argv, "t:m:")) != -1) {
        switch (opt) {
        case 't':
            num_threads = std::stoi(optarg);
            break;
        case 'm':
            num_models = std::stoi(optarg);
            break;
        default:
            std::cerr << "Usage: " << argv[0] << " -t <threads> -m <models>" << std::endl;
            exit(EXIT_FAILURE);
        }
    }

#if defined(_OPENMP) || defined(_MPI)
    omp_set_num_threads(num_threads);
#endif

#ifdef _MPI
	MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); 
    MPI_Comm_size(MPI_COMM_WORLD, &size);
#endif

	Neural::Dataset data_learning;
	data_learning.loadInputOutputData(4, 3, "database/iris.txt");

	vector<vector<double>> input = data_learning.getInput();
	vector<vector<double>> output = data_learning.getOutput();

	int max_epochs = 1000;
	int desired_hit_percent = 95;
	double error_tolerance = 0.05;
	int hidden_layer_limit = 15;
	double learning_rate_increase = 0.25;

	Neural::Network neural_network(input, output);
    // Set parameters: max_epochs, desired_hit, error_tolerance, learning_rate, hidden_layer_size
	neural_network.setParameter(max_epochs, desired_hit_percent, error_tolerance, 0.25, 15);

	// Início da medição de tempo
	auto start_time = std::chrono::high_resolution_clock::now();

	#ifdef _SEQUENTIAL
	// neural_network.autoTraining(hidden_layer_limit, learning_rate_increase);
    // Sequential also needs to support trainModels if we want to test it
    // But for now, let's assume we are building for OpenMP/CUDA
    neural_network.trainModels(num_models, num_threads);
	#endif

	// Executa OpenMP puro (v=openmp)
	#ifdef _OPENMP
		#ifndef _MPI 
		neural_network.trainModels(num_models, num_threads);
		#endif
	#endif
	
    // CUDA (v=cuda)
    #ifdef _CUDA
        neural_network.trainModels(num_models, num_threads);
    #endif

	// Executa MPI 
	#ifdef _MPI
		neural_network.autoTrainingMPI(hidden_layer_limit, learning_rate_increase, rank, size);
	#endif

	// Fim da medição de tempo
	auto end_time = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed_time = end_time - start_time;

// Apenas o processo 0 imprime o tempo
#ifdef _MPI
	if (rank == 0) {
#endif
		std::cout << "Tempo de execução: " << elapsed_time.count() << " segundos" << std::endl;
#ifdef _MPI
	}
#endif

	#ifdef _MPI
	MPI_Finalize();
	#endif

	return 0;
}