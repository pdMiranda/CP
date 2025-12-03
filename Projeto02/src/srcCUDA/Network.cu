#include "../include/Network.hpp"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <ctime>

using namespace std;

namespace Neural {

// CUDA Error checking helper
#define cudaCheckError(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

// Kernels
__device__ double sigmoid_device(double z){
    return 1.0/(1.0+exp(-z));
}

__device__ double sigmoidPrime_device(double z){
    return exp(-z) / ( pow(1.0+exp(-z),2) );
}

__global__ void forwardHiddenKernel(double* input, double* weights, double* hidden_sum, double* hidden_act, 
                                    int input_size, int hidden_size, int data_row_offset) {
    int h = blockIdx.x * blockDim.x + threadIdx.x;
    if (h < hidden_size) {
        double sum = 0.0;
        // Input size includes bias? 
        // We assume input array has bias handled or we handle it here.
        // In Network.cpp we passed input_line with bias.
        // Here we might pass the whole input matrix.
        // Let's assume we process one row at a time for simplicity of porting, 
        // or we pass the row pointer.
        
        for (int i = 0; i < input_size; i++) {
            double in_val = (i == input_size - 1) ? 1.0 : input[data_row_offset + i];
            sum += in_val * weights[i * hidden_size + h];
        }
        hidden_sum[h] = sum;
        hidden_act[h] = sigmoid_device(sum);
    }
}

__global__ void forwardOutputKernel(double* hidden_act, double* weights, double* output_sum, double* output_act, 
                                    int hidden_size, int output_size) {
    int o = blockIdx.x * blockDim.x + threadIdx.x;
    if (o < output_size) {
        double sum = 0.0;
        for (int h = 0; h < hidden_size; h++) {
            sum += hidden_act[h] * weights[h * output_size + o];
        }
        output_sum[o] = sum;
        output_act[o] = sigmoid_device(sum);
    }
}

__global__ void backPropDeltaKernel(double* output_act, double* target, double* output_sum, double* delta_out, int output_size) {
    int o = blockIdx.x * blockDim.x + threadIdx.x;
    if (o < output_size) {
        delta_out[o] = (target[o] - output_act[o]) * sigmoidPrime_device(output_sum[o]);
    }
}

__global__ void backPropHiddenDeltaKernel(double* delta_out, double* weights_out, double* hidden_sum, double* delta_in, 
                                          int output_size, int hidden_size) {
    int h = blockIdx.x * blockDim.x + threadIdx.x;
    if (h < hidden_size) {
        double sum = 0.0;
        for (int o = 0; o < output_size; o++) {
            sum += delta_out[o] * weights_out[h * output_size + o];
        }
        delta_in[h] = sum * sigmoidPrime_device(hidden_sum[h]);
    }
}

__global__ void updateWeightsOutputKernel(double* weights, double* delta_out, double* hidden_act, 
                                          int hidden_size, int output_size, double learning_rate) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = hidden_size * output_size;
    if (idx < total) {
        int h = idx / output_size;
        int o = idx % output_size;
        weights[idx] += delta_out[o] * hidden_act[h] * learning_rate;
    }
}

__global__ void updateWeightsInputKernel(double* weights, double* delta_in, double* input, 
                                         int input_size, int hidden_size, double learning_rate, int data_row_offset) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = input_size * hidden_size;
    if (idx < total) {
        int i = idx / hidden_size;
        int h = idx % hidden_size;
        double in_val = (i == input_size - 1) ? 1.0 : input[data_row_offset + i];
        weights[idx] += delta_in[h] * in_val * learning_rate;
    }
}

// Network Implementation
Network::Network(){
}

Network::Network(vector<vector<double>> user_input, vector<vector<double>> user_output){
    setInput(user_input);
    setOutput(user_output);
    output_layer_size = user_output[0].size();
}

void Network::setParameter( int user_max_epoch, int user_desired_percent, double user_error_tolerance, double user_learning_rate, int user_hidden_layer_size){
    setMaxEpoch(user_max_epoch);
	setLearningRate(user_learning_rate);
    setErrorTolerance(user_error_tolerance);
    setDesiredPercent(user_desired_percent);
    setHiddenLayerSize(user_hidden_layer_size);
    best_network.epoch = max_epoch;
    initializeWeight();
}

void Network::trainModels(int num_models, int num_threads) {
    cout << "Training " << num_models << " models with CUDA..." << endl;
    
    // Allocate device memory once if possible
    double *d_input, *d_output, *d_w_in, *d_w_out;
    double *d_hidden_sum, *d_hidden_act, *d_out_sum, *d_out_act;
    double *d_delta_out, *d_delta_in;
    
    // Flattened input size (without bias in vector, but logic handles it)
    size_t input_bytes = input.size() * sizeof(double);
    size_t output_bytes = output.size() * sizeof(double);
    size_t w_in_bytes = input_layer_size * hidden_layer_size * sizeof(double);
    size_t w_out_bytes = hidden_layer_size * output_layer_size * sizeof(double);
    
    cudaCheckError(cudaMalloc(&d_input, input_bytes));
    cudaCheckError(cudaMalloc(&d_output, output_bytes));
    cudaCheckError(cudaMalloc(&d_w_in, w_in_bytes));
    cudaCheckError(cudaMalloc(&d_w_out, w_out_bytes));
    
    // Intermediate buffers (max size needed)
    cudaCheckError(cudaMalloc(&d_hidden_sum, hidden_layer_size * sizeof(double)));
    cudaCheckError(cudaMalloc(&d_hidden_act, hidden_layer_size * sizeof(double)));
    cudaCheckError(cudaMalloc(&d_out_sum, output_layer_size * sizeof(double)));
    cudaCheckError(cudaMalloc(&d_out_act, output_layer_size * sizeof(double)));
    cudaCheckError(cudaMalloc(&d_delta_out, output_layer_size * sizeof(double)));
    cudaCheckError(cudaMalloc(&d_delta_in, hidden_layer_size * sizeof(double)));
    
    // Copy input/output once
    cudaCheckError(cudaMemcpy(d_input, input.data(), input_bytes, cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(d_output, output.data(), output_bytes, cudaMemcpyHostToDevice));
    
    int blockSize = num_threads > 0 ? num_threads : 256;
    
    for (int m = 0; m < num_models; m++) {
        cout << "Model " << m + 1 << "/" << num_models << "..." << endl;
        initializeWeight();
        
        // Copy weights to device
        cudaCheckError(cudaMemcpy(d_w_in, weight_input.data(), w_in_bytes, cudaMemcpyHostToDevice));
        cudaCheckError(cudaMemcpy(d_w_out, weight_output.data(), w_out_bytes, cudaMemcpyHostToDevice));
        
        // Training Loop
        int num_rows = input.size() / (input_layer_size - 1);
        
        for (epoch = 0; epoch < max_epoch && hit_percent < desired_percent; epoch++) {
            for (int row = 0; row < num_rows; row++) {
                // Forward
                int row_offset = row * (input_layer_size - 1);
                
                // Hidden
                int grid_hidden = (hidden_layer_size + blockSize - 1) / blockSize;
                forwardHiddenKernel<<<grid_hidden, blockSize>>>(d_input, d_w_in, d_hidden_sum, d_hidden_act, 
                                                                input_layer_size, hidden_layer_size, row_offset);
                
                // Output
                int grid_out = (output_layer_size + blockSize - 1) / blockSize;
                forwardOutputKernel<<<grid_out, blockSize>>>(d_hidden_act, d_w_out, d_out_sum, d_out_act, 
                                                             hidden_layer_size, output_layer_size);
                
                // Backprop
                // Delta Out
                // Target offset
                double* d_target_row = d_output + row * output_layer_size;
                backPropDeltaKernel<<<grid_out, blockSize>>>(d_out_act, d_target_row, d_out_sum, d_delta_out, output_layer_size);
                
                // Delta Hidden
                backPropHiddenDeltaKernel<<<grid_hidden, blockSize>>>(d_delta_out, d_w_out, d_hidden_sum, d_delta_in, 
                                                                      output_layer_size, hidden_layer_size);
                
                // Update Weights Output
                int total_w_out = hidden_layer_size * output_layer_size;
                int grid_w_out = (total_w_out + blockSize - 1) / blockSize;
                updateWeightsOutputKernel<<<grid_w_out, blockSize>>>(d_w_out, d_delta_out, d_hidden_act, 
                                                                     hidden_layer_size, output_layer_size, learning_rate);
                
                // Update Weights Input
                int total_w_in = input_layer_size * hidden_layer_size;
                int grid_w_in = (total_w_in + blockSize - 1) / blockSize;
                updateWeightsInputKernel<<<grid_w_in, blockSize>>>(d_w_in, d_delta_in, d_input, 
                                                                   input_layer_size, hidden_layer_size, learning_rate, row_offset);
            }
            
            // Calculate Hit Rate (Run)
            // We can do this on device too, but let's just copy weights back or keep them there?
            // To calculate hit rate, we need to run forward on all data.
            // Let's implement a kernel for that or just reuse the forward kernels.
            // For simplicity, let's copy weights back to host and use the host run() logic?
            // No, that's slow. Let's do it on device.
            
            // Actually, let's just copy weights back once per epoch to check termination condition
            // Or implement a reduction kernel for hit count.
            
            // For now, let's copy weights back to host and use CPU run() to reuse logic and ensure correctness
            // It's slower but safer for implementation speed.
            cudaCheckError(cudaMemcpy(weight_input.data(), d_w_in, w_in_bytes, cudaMemcpyDeviceToHost));
            cudaCheckError(cudaMemcpy(weight_output.data(), d_w_out, w_out_bytes, cudaMemcpyDeviceToHost));
            
            run(); // Host run
        }
        
        cout << "Model " << m + 1 << " finished. Epochs: " << epoch << ", Hit %: " << hit_percent << endl;
    }
    
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_w_in);
    cudaFree(d_w_out);
    cudaFree(d_hidden_sum);
    cudaFree(d_hidden_act);
    cudaFree(d_out_sum);
    cudaFree(d_out_act);
    cudaFree(d_delta_out);
    cudaFree(d_delta_in);
}

void Network::run(){
    // Host implementation of run (reused from sequential logic)
    // Since we copied weights back in trainModels loop, this works.
    correct_output = 0;
    int num_rows = input.size() / (input_layer_size - 1);
    
    for (int data_row = 0; data_row < num_rows; data_row++){
        vector<double> input_line;
        int base_idx = data_row * (input_layer_size - 1);
        for(int k=0; k<input_layer_size-1; k++) input_line.push_back(input[base_idx+k]);
        
        ForwardPropagation forward = forwardPropagation(input_line);
        
        // Check correctness
        bool correct = true;
        int out_base = data_row * output_layer_size;
        for(int i=0; i<output_layer_size; i++) {
             if (abs(forward.output[i] - output[out_base+i]) >= error_tolerance) {
                 correct = false;
                 break;
             }
        }
        if(correct) correct_output++;
    }
    hitRateCalculate();    
}

// Host helper methods (same as before)
Network::ForwardPropagation Network::forwardPropagation(vector<double> input_line){
    input_line.push_back(1); 
    ForwardPropagation forward(hidden_layer_size, output_layer_size);

    for (int i = 0; i < hidden_layer_size; i++ ){
        for (int j = 0; j < input_layer_size; j++ ){
            forward.sum_input_weight[i] += input_line[j] * weight_input[j * hidden_layer_size + i];
        }
    }
    for (int i = 0; i < hidden_layer_size; i++ )
        forward.sum_input_weight_ativation.push_back(sigmoid(forward.sum_input_weight[i]));

    for (int i = 0; i < output_layer_size; i++ ){
        for (int j = 0; j < hidden_layer_size; j++ ){
            forward.sum_output_weigth[i] += forward.sum_input_weight_ativation[j] * weight_output[j * output_layer_size + i];
        }
    }
    for (int i = 0; i < output_layer_size; i++ )
        forward.output.push_back(sigmoid(forward.sum_output_weigth[i]));

    return forward;
}

// Unused in CUDA training but needed for compilation
void Network::backPropagation(ForwardPropagation, vector<double>, vector<double>){}
void Network::trainingClassification(){}
void Network::autoTraining(int, double){}
void Network::trainingTemporal(){}

void Network::hitRateCalculate(){
    hit_percent = (correct_output*100) / (input.size() / (input_layer_size - 1));
}

void Network::initializeWeight(){
    weight_input.resize(input_layer_size * hidden_layer_size);
    weight_output.resize(hidden_layer_size * output_layer_size);
    
    for(int i=0; i<weight_input.size(); i++) weight_input[i] = ((double) rand() / (RAND_MAX));
    for(int i=0; i<weight_output.size(); i++) weight_output[i] = ((double) rand() / (RAND_MAX));

    hit_percent = 0;
    correct_output = 0;
}

double Network::sigmoid(double z){ return 1/(1+exp(-z)); }	
double Network::sigmoidPrime(double z){ return exp(-z) / ( pow(1+exp(-z),2) ); }

void Network::setMaxEpoch(int m){ max_epoch = m; }
void Network::setDesiredPercent(int d){ desired_percent = d; }
void Network::setHiddenLayerSize(int h){ hidden_layer_size = h; }
void Network::setLearningRate(double l){ learning_rate = l; }
void Network::setErrorTolerance(double e){ error_tolerance = e; }

void Network::setInput(vector<vector<double>> i){
    input_layer_size = i[0].size() + 1; 
    input.clear();
    for(const auto& row : i) for(double val : row) input.push_back(val);
}

void Network::setOutput(vector<vector<double>> o){
    output_layer_size = o[0].size();
    output.clear();
    for(const auto& row : o) for(double val : row) output.push_back(val);
}

}
