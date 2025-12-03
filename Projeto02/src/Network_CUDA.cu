#include "../include/Network.hpp"
#include <cuda_runtime.h>
#include <cstdio>

namespace Neural {

__device__ double d_sigmoid(double z) { return 1.0 / (1.0 + exp(-z)); }
__device__ double d_sigmoidPrime(double sig_val) { return sig_val * (1.0 - sig_val); }

// Kernel: 1 Bloco = 1 Modelo Neural
// Threads dentro do bloco colaboram (ou apenas a thread 0 faz o trabalho sequencial simples para SGD)
// Para simplificar e evitar race conditions complexas no backprop, faremos cada Thread treinar 1 Modelo (Grade massiva) 
// OU Bloco faz o modelo (ideal para matrizes grandes, mas Iris é pequeno).
// DADO QUE IRIS É PEQUENO: Vamos fazer 1 Thread = 1 Modelo para maximizar ocupação com "num_models" alto.
__global__ void train_kernel(
    double* input, double* output, 
    double* all_w_in, double* all_w_out,
    int n_rows, int n_in, int n_out, int hidden_size,
    int epochs, double lr, int num_models
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_models) return;

    // Offset dos pesos para este modelo específico
    double* w_in = &all_w_in[tid * (n_in * hidden_size)];
    double* w_out = &all_w_out[tid * (hidden_size * n_out)];

    // Arrays locais (registradores/lmem)
    double hidden_acts[20];
    double output_acts[5];
    double delta_out[5];
    double delta_hidden[20];

    for (int e = 0; e < epochs; e++) {
        for (int row = 0; row < n_rows; row++) {
            
            // Forward Input->Hidden
            for (int h = 0; h < hidden_size; h++) {
                double sum = 0.0;
                for (int i = 0; i < n_in; i++) {
                    sum += input[row * n_in + i] * w_in[i * hidden_size + h];
                }
                hidden_acts[h] = d_sigmoid(sum);
            }

            // Forward Hidden->Output
            for (int o = 0; o < n_out; o++) {
                double sum = 0.0;
                for (int h = 0; h < hidden_size; h++) {
                    sum += hidden_acts[h] * w_out[h * n_out + o];
                }
                output_acts[o] = d_sigmoid(sum);
            }

            // Backprop Errors
            for (int o = 0; o < n_out; o++) {
                double err = output[row * n_out + o] - output_acts[o];
                delta_out[o] = err * d_sigmoidPrime(output_acts[o]);
            }

            for (int h = 0; h < hidden_size; h++) {
                double err = 0.0;
                for (int o = 0; o < n_out; o++) {
                    err += delta_out[o] * w_out[h * n_out + o];
                }
                delta_hidden[h] = err * d_sigmoidPrime(hidden_acts[h]);
            }

            // Updates
            for (int h = 0; h < hidden_size; h++) {
                for (int o = 0; o < n_out; o++) {
                    w_out[h * n_out + o] += lr * delta_out[o] * hidden_acts[h];
                }
            }
            for (int i = 0; i < n_in; i++) {
                for (int h = 0; h < hidden_size; h++) {
                    w_in[i * hidden_size + h] += lr * delta_hidden[h] * input[row * n_in + i];
                }
            }
        }
    }
}

void Network::trainCUDA(int num_models, int num_blocks, int num_threads) {
    std::cout << ">>> Iniciando Treinamento CUDA" << std::endl;
    
    int hidden_size = 6;
    int epochs = 1000;
    double lr = 0.1;

    size_t size_in = n_rows * n_cols_in * sizeof(double);
    size_t size_out = n_rows * n_cols_out * sizeof(double);
    size_t size_w_in = num_models * n_cols_in * hidden_size * sizeof(double);
    size_t size_w_out = num_models * hidden_size * n_cols_out * sizeof(double);

    double *d_input, *d_output, *d_w_in, *d_w_out;
    double *h_w_in = new double[num_models * n_cols_in * hidden_size];
    double *h_w_out = new double[num_models * hidden_size * n_cols_out];

    // Inicializa pesos
    for(int i=0; i<num_models * n_cols_in * hidden_size; i++) h_w_in[i] = (double)rand()/RAND_MAX;
    for(int i=0; i<num_models * hidden_size * n_cols_out; i++) h_w_out[i] = (double)rand()/RAND_MAX;

    cudaMalloc(&d_input, size_in);
    cudaMalloc(&d_output, size_out);
    cudaMalloc(&d_w_in, size_w_in);
    cudaMalloc(&d_w_out, size_w_out);

    cudaMemcpy(d_input, h_input, size_in, cudaMemcpyHostToDevice);
    cudaMemcpy(d_output, h_output, size_out, cudaMemcpyHostToDevice);
    cudaMemcpy(d_w_in, h_w_in, size_w_in, cudaMemcpyHostToDevice);
    cudaMemcpy(d_w_out, h_w_out, size_w_out, cudaMemcpyHostToDevice);

    // Configuração de Kernel: Total de threads deve cobrir o número de modelos
    // Se num_threads > 1, usamos mais threads por bloco para processar mais modelos
    int threads_per_block = num_threads; 
    int blocks = (num_models + threads_per_block - 1) / threads_per_block;

    // Override com o parâmetro se fornecido explicitamente, mas garantindo cobertura
    if (num_blocks > 0) blocks = num_blocks; 

    std::cout << "Config CUDA: Blocks=" << blocks << ", Threads=" << threads_per_block << std::endl;

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    train_kernel<<<blocks, threads_per_block>>>(
        d_input, d_output, d_w_in, d_w_out,
        n_rows, n_cols_in, n_cols_out, hidden_size,
        epochs, lr, num_models
    );
    cudaEventRecord(stop);
    
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    std::cout << "Tempo Total CUDA: " << milliseconds/1000.0 << "s" << std::endl;

    cudaFree(d_input); cudaFree(d_output); cudaFree(d_w_in); cudaFree(d_w_out);
    delete[] h_w_in; delete[] h_w_out;
}

}