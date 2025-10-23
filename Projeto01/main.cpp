#include "include/Network.hpp"
#include "include/Dataset.hpp"
#include <chrono> // Para medir o tempo de execução
#include <omp.h> // Para configurar o número de threads
#ifdef _MPI
#include <mpi.h>
#endif

int main(int argc, char* argv[])
{
#ifdef _OPENMP
    omp_set_num_threads(NUM_THREADS);
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

	int max_epochs = 1000;				  // numero maximo de epocas que serao testadas
	int desired_hit_percent = 95;		  // numero minimo de porcentagem de acertos aceitados
	double error_tolerance = 0.05;		  // tolerancia de erro. se for maior que ele, e considerado como previsao errada
	int hidden_layer_limit = 15;		  // numero maximo de camadas escondidas
	double learning_rate_increase = 0.25; // aumento da taxa de aprendizado (quanto em quanto, de 0.0 a 1.0)

	Neural::Network neural_network(input, output);
	neural_network.setParameter(max_epochs, desired_hit_percent, error_tolerance);

	#ifdef _MPI
		neural_network.setParameter(max_epochs, desired_hit_percent, error_tolerance, 0.1, 3);
	#endif
	// Início da medição de tempo
	auto start_time = std::chrono::high_resolution_clock::now();

	#ifdef _SEQUENTIAL
	neural_network.autoTraining(hidden_layer_limit, learning_rate_increase);
	#endif

	#ifdef _OPENMP
		neural_network.autoTraining(hidden_layer_limit, learning_rate_increase);
	#endif
	
	#ifdef _MPI
		neural_network.autoTrainingMPI(hidden_layer_limit, learning_rate_increase, rank, size);
	#endif

	// Fim da medição de tempo
	auto end_time = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed_time = end_time - start_time;

	// Exibe o tempo de execução
	std::cout << "Tempo de execução: " << elapsed_time.count() << " segundos" << std::endl;

	#ifdef _MPI
	MPI_Finalize();
	#endif

	return 0;
}