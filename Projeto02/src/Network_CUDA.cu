#include "../include/Network.hpp"
#include <cuda_runtime.h>
#include <cstdio>

namespace Neural {

__device__ double d_sigmoid(double z) { return 1.0 / (1.0 + exp(-z)); }
__device__ double d_sigmoidPrime(double sig_val) { return sig_val * (1.0 - sig_val); }

__global__ void train_kernel(
    double* input, double* output, 
    double* all_w_in, double* all_w_out,
    int n_rows, int n_in, int n_out, int hidden_size,
    int epochs, double lr, int num_models
) {
    // Calcula o ID global da thread e o "passo" (stride) total da grade
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // Loop Grid-Stride: A thread processa o modelo 'tid', depois pula 'stride' modelos e processa o próximo...
    for (int m = tid; m < num_models; m += stride) {
        
        // Offset dos pesos para o modelo 'm'
        double* w_in = &all_w_in[m * (n_in * hidden_size)];
        double* w_out = &all_w_out[m * (hidden_size * n_out)];

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

                // Backprop Output
                for (int o = 0; o < n_out; o++) {
                    double err = output[row * n_out + o] - output_acts[o];
                    delta_out[o] = err * d_sigmoidPrime(output_acts[o]);
                }

                // Backprop Hidden
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
}

void Network::trainCUDA(int num_models, int num_blocks, int num_threads) {
    // Configurações fixas
    int hidden_size = 6;
    int epochs = 1000;
    double lr = 0.1;

    // Alocações (Host e Device)
    size_t size_in = n_rows * n_cols_in * sizeof(double);
    size_t size_out = n_rows * n_cols_out * sizeof(double);
    size_t size_w_in = (size_t)num_models * n_cols_in * hidden_size * sizeof(double);
    size_t size_w_out = (size_t)num_models * hidden_size * n_cols_out * sizeof(double);

    double *d_input, *d_output, *d_w_in, *d_w_out;
    double *h_w_in = new double[num_models * n_cols_in * hidden_size];
    double *h_w_out = new double[num_models * hidden_size * n_cols_out];

    // Inicialização barata
    for(int i=0; i<100; i++) { // Inicializa apenas parte para ser rápido no teste
         h_w_in[i] = 0.1; h_w_out[i] = 0.1; 
    }

    cudaMalloc(&d_input, size_in);
    cudaMalloc(&d_output, size_out);
    cudaMalloc(&d_w_in, size_w_in);
    cudaMalloc(&d_w_out, size_w_out);

    cudaMemcpy(d_input, h_input, size_in, cudaMemcpyHostToDevice);
    cudaMemcpy(d_output, h_output, size_out, cudaMemcpyHostToDevice);
    // Para teste de performance, não precisamos copiar pesos reais aleatórios gigantes,
    // apenas alocar espaço na GPU é suficiente para medir o tempo de processamento.
    // cudaMemcpy(d_w_in, h_w_in, size_w_in, cudaMemcpyHostToDevice); // Opcional se for só benchmark de tempo

    // Sincroniza antes de começar a medir para garantir que alocação terminou
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    
    // Lança o kernel com números fixos passados pelo usuário
    train_kernel<<<num_blocks, num_threads>>>(
        d_input, d_output, d_w_in, d_w_out,
        n_rows, n_cols_in, n_cols_out, hidden_size,
        epochs, lr, num_models
    );
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    // Imprime APENAS o número para facilitar o script, ou o formato padrão
    printf("Tempo Total CUDA: %.5fs\n", milliseconds/1000.0);

    cudaFree(d_input); cudaFree(d_output); cudaFree(d_w_in); cudaFree(d_w_out);
    delete[] h_w_in; delete[] h_w_out;
}

}