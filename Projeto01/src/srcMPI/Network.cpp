#include "../include/Network.hpp"
#include <mpi.h>
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

// ===================== Backpropagation =====================
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

    for (unsigned int i = 0; i < weight_output.size(); i++)
        for (unsigned int j = 0; j < weight_output[i].size(); j++)
            weight_output[i][j] += back.delta_output_sum[j] * forward.sum_input_weight_ativation[i] * learning_rate;

    for (unsigned int i = 0; i < weight_input.size(); i++)
        for (unsigned int j = 0; j < weight_input[i].size(); j++)
            weight_input[i][j] += back.delta_input_sum[j] * input_line[i] * learning_rate;
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

// ===================== Inicialização =====================
void Network::initializeWeight() {
    weight_input.resize(input_layer_size);
    for (int i = 0; i < input_layer_size; i++)
        weight_input[i].resize(hidden_layer_size);

    weight_output.resize(hidden_layer_size);
    for (int i = 0; i < hidden_layer_size; i++)
        weight_output[i].resize(output_layer_size);

    srand((unsigned int)time(0));
    for (int i = 0; i < input_layer_size; i++)
        for (int j = 0; j < hidden_layer_size; j++)
            weight_input[i][j] = ((double)rand() / RAND_MAX);

    for (int i = 0; i < hidden_layer_size; i++)
        for (int j = 0; j < output_layer_size; j++)
            weight_output[i][j] = ((double)rand() / RAND_MAX);

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

// ===================== Run MPI =====================
void Network::run(int rank, int size) {
    int start = (input.size() * rank) / size;
    int end = (input.size() * (rank + 1)) / size;

    int local_correct = 0;
    for (int i = start; i < end; i++) {
        ForwardPropagation forward = forwardPropagation(input[i]);
        for (int j = 0; j < output_layer_size; j++)
            if (abs(forward.output[j] - output[i][j]) < error_tolerance)
                local_correct++;
    }

    int global_correct = 0;
    MPI_Reduce(&local_correct, &global_correct, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0)
        hit_percent = (global_correct * 100) / (output.size() * output_layer_size);
}

// ===================== Treinamento MPI =====================
void Network::trainingClassification(int rank, int size) {
    for (epoch = 0; epoch < max_epoch && hit_percent < desired_percent; epoch++) {
        int start = (input.size() * rank) / size;
        int end = (input.size() * (rank + 1)) / size;

        vector<vector<double>> local_weight_input = weight_input;
        vector<vector<double>> local_weight_output = weight_output;

        for (int i = start; i < end; i++) {
            ForwardPropagation forward = forwardPropagation(input[i]);
            backPropagation(forward, input[i], output[i]);
        }

        // MPI_Allreduce: soma os pesos
        for (int i = 0; i < input_layer_size; i++)
            MPI_Allreduce(MPI_IN_PLACE, weight_input[i].data(), hidden_layer_size, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        for (int i = 0; i < hidden_layer_size; i++)
            MPI_Allreduce(MPI_IN_PLACE, weight_output[i].data(), output_layer_size, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        run(rank, size);

        //if (rank == 0)
           // cout << "Hidden Layer Size: " << hidden_layer_size 
           //     << "\tLearning Rate: " << learning_rate 
            //    << "\tHit Percent: " << hit_percent << "%" 
            //    << "\tEpoch: " << epoch << endl;
    }
}

// ===================== Auto Training MPI =====================
void Network::autoTrainingMPI(int hidden_layer_limit, double learning_rate_increase, int rank, int size) {

    cout << "Iniciando auto-treinamento paralelo em " << rank << " threads e com "<< size << " processos..."<< endl;

    for (hidden_layer_size = 3; hidden_layer_size <= hidden_layer_limit; hidden_layer_size++) {
        for (learning_rate = learning_rate_increase; learning_rate <= 1.0; learning_rate += learning_rate_increase) {
            initializeWeight();
            trainingClassification(rank, size);

            if (rank == 0 && epoch < best_network.epoch) {
                best_network.epoch = epoch;
                best_network.learning_rate = learning_rate;
                best_network.hidden_layer = hidden_layer_size;
                best_network.weight_input = weight_input;
                best_network.weight_output = weight_output;
            }

            // Broadcast para todos os processos
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
