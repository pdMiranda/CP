#include "../include/Network.hpp"
#include <omp.h>
#include <cstdlib>
#include <cstdio>

namespace Neural {

#pragma omp declare target
double Network::sigmoid(double z) { return 1.0 / (1.0 + exp(-z)); }
double Network::sigmoidPrime(double z) { double t = sigmoid(z); return t * (1.0 - t); }
#pragma omp end declare target

void Network::trainOpenMP(int num_models, int num_teams, int num_threads) {
    std::cout << ">>> Iniciando Treinamento OpenMP GPU" << std::endl;
    std::cout << "Modelos: " << num_models << " | Teams: " << num_teams << " | Threads/Team: " << num_threads << std::endl;

    int total_input_size = n_rows * n_cols_in;
    int total_output_size = n_rows * n_cols_out;
    
    // Configurações fixas para teste de escalabilidade (poderiam ser aleatórias)
    int hidden_size = 6;
    int epochs = 1000;
    double lr = 0.1;

    // Tamanho dos pesos para UM modelo
    int w_in_size = n_cols_in * hidden_size; // input -> hidden
    int w_out_size = hidden_size * n_cols_out; // hidden -> output

    // Alocação total para todos os modelos (Pesos são independentes)
    int total_weights_in = w_in_size * num_models;
    int total_weights_out = w_out_size * num_models;

    double* all_w_in = new double[total_weights_in];
    double* all_w_out = new double[total_weights_out];

    // Inicialização aleatória no host
    for(int i=0; i<total_weights_in; i++) all_w_in[i] = ((double)rand() / RAND_MAX);
    for(int i=0; i<total_weights_out; i++) all_w_out[i] = ((double)rand() / RAND_MAX);

    double start_time = omp_get_wtime();

    // Mapeamento de dados para a GPU
    #pragma omp target data map(to: h_input[0:total_input_size], h_output[0:total_output_size]) \
                            map(tofrom: all_w_in[0:total_weights_in], all_w_out[0:total_weights_out])
    {
        // Paralelismo de GRÃO GROSSO: Cada TEAM treina UM MODELO
        #pragma omp target teams distribute parallel for num_teams(num_teams) thread_limit(num_threads)
        for (int m = 0; m < num_models; m++) {
            
            // Ponteiros locais para os pesos deste modelo específico
            double* my_w_in = &all_w_in[m * w_in_size];
            double* my_w_out = &all_w_out[m * w_out_size];

            // Variáveis locais da thread (privatização)
            // Nota: Em OpenMP GPU, alocação dinâmica dentro do kernel é ruim.
            // Usamos arrays estáticos pequenos assumindo limites do problema Iris.
            double hidden_acts[20]; // Max 20 neurônios ocultos
            double output_acts[5];  // Max 5 saídas

            for (int epoch = 0; epoch < epochs; epoch++) {
                
                // Stochastic Gradient Descent (SGD) - Amostra por amostra
                for (int row = 0; row < n_rows; row++) {
                    
                    // --- Forward ---
                    // Input -> Hidden
                    for (int h = 0; h < hidden_size; h++) {
                        double sum = 0.0;
                        for (int i = 0; i < n_cols_in; i++) {
                            sum += h_input[row * n_cols_in + i] * my_w_in[i * hidden_size + h];
                        }
                        hidden_acts[h] = sigmoid(sum);
                    }

                    // Hidden -> Output
                    for (int o = 0; o < n_cols_out; o++) {
                        double sum = 0.0;
                        for (int h = 0; h < hidden_size; h++) {
                            sum += hidden_acts[h] * my_w_out[h * n_cols_out + o];
                        }
                        output_acts[o] = sigmoid(sum);
                    }

                    // --- Backward ---
                    // Output Error
                    double delta_out[5];
                    for (int o = 0; o < n_cols_out; o++) {
                        double error = h_output[row * n_cols_out + o] - output_acts[o];
                        delta_out[o] = error * sigmoidPrime(output_acts[o]); // Aproximação derivada
                    }

                    // Hidden Error
                    double delta_hidden[20];
                    for (int h = 0; h < hidden_size; h++) {
                        double error = 0.0;
                        for (int o = 0; o < n_cols_out; o++) {
                            error += delta_out[o] * my_w_out[h * n_cols_out + o];
                        }
                        delta_hidden[h] = error * sigmoidPrime(hidden_acts[h]); // Aproximação
                    }

                    // Update Weights (Hidden -> Output)
                    for (int h = 0; h < hidden_size; h++) {
                        for (int o = 0; o < n_cols_out; o++) {
                            my_w_out[h * n_cols_out + o] += lr * delta_out[o] * hidden_acts[h];
                        }
                    }

                    // Update Weights (Input -> Hidden)
                    for (int i = 0; i < n_cols_in; i++) {
                        for (int h = 0; h < hidden_size; h++) {
                            my_w_in[i * hidden_size + h] += lr * delta_hidden[h] * h_input[row * n_cols_in + i];
                        }
                    }
                }
            }
        }
    }

    double end_time = omp_get_wtime();
    std::cout << "Tempo Total OpenMP: " << (end_time - start_time) << "s" << std::endl;

    delete[] all_w_in;
    delete[] all_w_out;
}

}