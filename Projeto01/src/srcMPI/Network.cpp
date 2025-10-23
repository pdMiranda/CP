#include "../include/Network.hpp"
#include <mpi.h>
#include <omp.h>
#include <cmath>
#include <ctime>
#include <cstdlib>
#include <iostream>

using namespace std;

namespace Neural {

// ===================== Construtores =====================
Network::Network() { }

Network::Network(vector<vector<double>> user_input, vector<vector<double>> user_output) {
    setInput(user_input);
    setOutput(user_output);
    output_layer_size = 3;
}

// ===================== Configuração =====================
void Network::setParameter(int user_max_epoch, int user_desired_percent, double user_error_tolerance, double user_learning_rate, int user_hidden_layer_size) {
    setMaxEpoch(user_max_epoch);
    setLearningRate(user_learning_rate);
    setErrorTolerance(user_error_tolerance);
    setDesiredPercent(user_desired_percent);
    setHiddenLayerSize(user_hidden_layer_size);
    best_network.epoch = max_epoch;
    initializeWeight();
}

// ===================== Forward Propagation =====================
Network::ForwardPropagation Network::forwardPropagation(vector<double> input_line) {
    input_line.push_back(1.0); // bias
    ForwardPropagation forward(hidden_layer_size, output_layer_size);

    for (int i = 0; i < hidden_layer_size; i++)
        for (int j = 0; j < input_layer_size; j++)
            forward.sum_input_weight[i] += input_line[j] * weight_input[j][i];

    for (int i = 0; i < hidden_layer_size; i++)
        forward.sum_input_weight_ativation.push_back(sigmoid(forward.sum_input_weight[i]));

    for (int i = 0; i < output_layer_size; i++)
        for (int j = 0; j < hidden_layer_size; j++)
            forward.sum_output_weigth[i] += forward.sum_input_weight_ativation[j] * weight_output[j][i];

    for (int i = 0; i < output_layer_size; i++)
        forward.output.push_back(sigmoid(forward.sum_output_weigth[i]));

    return forward;
}

// ===================== Backpropagation (Thread-Safe) =====================
void Network::backPropagation(ForwardPropagation forward, vector<double> input_line, vector<double> output_line) {
    input_line.push_back(1.0); // bias
    BackPropagation back(hidden_layer_size);

    for (int i = 0; i < output_layer_size; i++)
        back.delta_output_sum.push_back((output_line[i] - forward.output[i]) * sigmoidPrime(forward.sum_output_weigth[i]));

    for (int i = 0; i < hidden_layer_size; i++) {
        for (int j = 0; j < output_layer_size; j++)
            back.delta_input_sum[i] += back.delta_output_sum[j] * weight_output[i][j];
        back.delta_input_sum[i] *= sigmoidPrime(forward.sum_input_weight[i]);
    }

    // Atualizações de peso agora são atômicas para segurança em OpenMP
    for (unsigned int i = 0; i < weight_output.size(); i++)
        for (unsigned int j = 0; j < weight_output[i].size(); j++){
            double delta = back.delta_output_sum[j] * forward.sum_input_weight_ativation[i] * learning_rate;
            #pragma omp atomic
            weight_output[i][j] += delta;
        }

    for (unsigned int i = 0; i < weight_input.size(); i++)
        for (unsigned int j = 0; j < weight_input[i].size(); j++){
            double delta = back.delta_input_sum[j] * input_line[i] * learning_rate;
            #pragma omp atomic
            weight_input[i][j] += delta;
        }
}

// ===================== Hit Rate =====================
void Network::hitRateCount(vector<double> neural_output, unsigned int data_row) {
    for (int i = 0; i < output_layer_size; i++)
        if (abs(neural_output[i] - output[data_row][i]) < error_tolerance)
            correct_output++;
}

void Network::hitRateCalculate() {
    hit_percent = (correct_output * 100) / (output.size() * output_layer_size);
    correct_output = 0;
}

// ===================== Inicialização (Sincronizada com MPI) =====================
void Network::initializeWeight() {
    weight_input.resize(input_layer_size);
    for (int i = 0; i < input_layer_size; i++)
        weight_input[i].resize(hidden_layer_size);

    weight_output.resize(hidden_layer_size);
    for (int i = 0; i < hidden_layer_size; i++)
        weight_output[i].resize(output_layer_size);

    // Garante que todos os processos MPI comecem com os mesmos pesos
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0) {
        srand((unsigned int)time(0)); // Apenas Rank 0 gera os pesos
        for (int i = 0; i < input_layer_size; i++)
            for (int j = 0; j < hidden_layer_size; j++)
                weight_input[i][j] = ((double)rand() / RAND_MAX);

        for (int i = 0; i < hidden_layer_size; i++)
            for (int j = 0; j < output_layer_size; j++)
                weight_output[i][j] = ((double)rand() / RAND_MAX);
    }

    // Rank 0 transmite (Broadcast) os pesos para todos os outros processos
    for (int i = 0; i < input_layer_size; i++)
        MPI_Bcast(weight_input[i].data(), hidden_layer_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    for (int i = 0; i < hidden_layer_size; i++)
        MPI_Bcast(weight_output[i].data(), output_layer_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);


    hit_percent = 0;
    correct_output = 0;
}

// ===================== Funções de ativação =====================
double Network::sigmoid(double z) { return 1 / (1 + exp(-z)); }
double Network::sigmoidPrime(double z) { return exp(-z) / pow(1 + exp(-z), 2); }

// ===================== Setters =====================
void Network::setMaxEpoch(int m) { max_epoch = m; }
void Network::setDesiredPercent(int d) { desired_percent = d; }
void Network::setHiddenLayerSize(int h) { hidden_layer_size = h; }
void Network::setLearningRate(double l) { learning_rate = l; }
void Network::setErrorTolerance(double e) { error_tolerance = e; }
void Network::setInput(vector<vector<double>> i) { input = i; input_layer_size = i[0].size() + 1; }
void Network::setOutput(vector<vector<double>> o) { output = o; output_layer_size = o[0].size(); }

// ===================== Run Híbrido (MPI + OpenMP) =====================
void Network::run(int rank, int size) {
    // Cada processo MPI cuida de um "chunk"
    int start = (input.size() * rank) / size;
    int end = (input.size() * (rank + 1)) / size;

    int local_correct = 0;
    
    // OpenMP paraleliza o loop sobre o "chunk" do processo
    #pragma omp parallel for reduction(+:local_correct)
    for (int i = start; i < end; i++) {
        ForwardPropagation forward = forwardPropagation(input[i]);
        for (int j = 0; j < output_layer_size; j++)
            if (abs(forward.output[j] - output[i][j]) < error_tolerance)
                local_correct++;
    }

    // MPI Reduz os contadores locais de todos os processos
    int global_correct = 0;
    MPI_Reduce(&local_correct, &global_correct, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    
    if (rank == 0)
        hit_percent = (global_correct * 100) / (output.size() * output_layer_size);
}

// ===================== Treinamento MPI =====================
void Network::trainingClassification(int rank, int size) {
    for (epoch = 0; epoch < max_epoch && hit_percent < desired_percent; epoch++) {
        // Cada processo MPI cuida de um "chunk"
        int start = (input.size() * rank) / size;
        int end = (input.size() * (rank + 1)) / size;

        // OpenMP paraleliza o loop de backpropagation sobre o "chunk"
        // (A função backPropagation foi tornada thread-safe com 'atomic')
        #pragma omp parallel for
        for (int i = start; i < end; i++) {
            ForwardPropagation forward = forwardPropagation(input[i]);
            backPropagation(forward, input[i], output[i]);
        }

        // MPI_Allreduce: soma os deltas dos pesos de todos os processos
        for (int i = 0; i < input_layer_size; i++)
            MPI_Allreduce(MPI_IN_PLACE, weight_input[i].data(), hidden_layer_size, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        for (int i = 0; i < hidden_layer_size; i++)
            MPI_Allreduce(MPI_IN_PLACE, weight_output[i].data(), output_layer_size, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        // run() agora é a versão híbrida
        run(rank, size);
    }
}

// ===================== Auto Training MPI =====================
void Network::autoTrainingMPI(int hidden_layer_limit, double learning_rate_increase, int rank, int size) {

    if (rank == 0) { // Apenas Rank 0 imprime
        cout << "Iniciando auto-treinamento Processos: " << size << ", Threads por Processo: " << omp_get_max_threads() << endl;
    }

    for (hidden_layer_size = 3; hidden_layer_size <= hidden_layer_limit; hidden_layer_size++) {
        for (learning_rate = learning_rate_increase; learning_rate <= 1.0; learning_rate += learning_rate_increase) {

            initializeWeight(); 

            trainingClassification(rank, size);

            if (rank == 0){
                cout << "Hidden Layer Size: " << hidden_layer_size 
                    << "\tLearning Rate: " << learning_rate 
                    << "\tHit Percent: " << hit_percent << "%" 
                    << "\tEpoch: " << epoch << endl;
            }

            if (rank == 0 && epoch < best_network.epoch) {
                best_network.epoch = epoch;
                best_network.learning_rate = learning_rate;
                best_network.hidden_layer = hidden_layer_size;
                best_network.weight_input = weight_input;
                best_network.weight_output = weight_output;
            }

            // Sincroniza os pesos (embora Allreduce já deva ter feito isso,
            // esta é a lógica original. Vamos mantê-la por segurança).
            for (int i = 0; i < input_layer_size; i++)
                MPI_Bcast(weight_input[i].data(), hidden_layer_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

            for (int i = 0; i < hidden_layer_size; i++)
                MPI_Bcast(weight_output[i].data(), output_layer_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        }
    }

    if (rank == 0) {
        epoch = best_network.epoch;
        learning_rate = best_network.learning_rate;
        hidden_layer_size = best_network.hidden_layer;
        weight_input = best_network.weight_input;
        weight_output = best_network.weight_output;

        cout << "\n-----------------------------------------------------" << endl;
        cout << "Treinamento finalizado." << endl;
        cout << "Melhor Rede Encontrada --> Hidden Layer: " << best_network.hidden_layer 
            << "\tLearning Rate: " << best_network.learning_rate 
            << "\tEpoch: " << best_network.epoch << endl;
        cout << "-----------------------------------------------------" << endl;
    }
}

} // namespace Neural