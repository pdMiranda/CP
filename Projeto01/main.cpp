#include "include/Network.hpp"
#include "include/Dataset.hpp"
#include <chrono> // Para medir o tempo de execução

int main()
{

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

	// Início da medição de tempo
	auto start_time = std::chrono::high_resolution_clock::now();

	neural_network.autoTraining(hidden_layer_limit, learning_rate_increase);

	// Fim da medição de tempo
	auto end_time = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed_time = end_time - start_time;

	// Exibe o tempo de execução
	std::cout << "Tempo de execução: " << elapsed_time.count() << " segundos" << std::endl;

	return 0;
}