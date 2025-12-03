#include "../include/Network.hpp"
#include <omp.h>
#include <cstring>
#include <cmath>

#ifndef NUM_THREADS
#define NUM_THREADS omp_get_max_threads()
#endif

namespace Neural{

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
    cout << "Training " << num_models << " models with " << num_threads << " threads (CPU Parallelism)..." << endl;
    
    // Set number of threads for the parallel region
    omp_set_num_threads(num_threads);

    #pragma omp parallel for
    for (int m = 0; m < num_models; m++) {
        // Create a thread-local copy of the network to avoid race conditions
        Network thread_net = *this;
        
        // Initialize weights randomly for this copy
        // We need to ensure different seeds.
        // initializeWeight uses srand(time(0)), which is not thread-safe or unique per thread if called fast.
        // We should modify initializeWeight or set seed here.
        // But initializeWeight is called inside.
        // Let's rely on the fact that we modified initializeWeight in previous steps to use thread id?
        // Wait, I overwrote the file. I need to check initializeWeight implementation below.
        
        thread_net.initializeWeight();
        thread_net.trainingClassification();
        
        #pragma omp critical
        {
            cout << "Model " << m + 1 << " finished (Thread " << omp_get_thread_num() << "). Epochs: " << thread_net.epoch << ", Hit %: " << thread_net.hit_percent << endl;
        }
    }
}

void Network::run(){
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

void Network::trainingClassification(){
    int num_rows = input.size() / (input_layer_size - 1);

    for (epoch = 0; epoch < max_epoch && hit_percent < desired_percent; epoch++) {
        for (int data_row = 0; data_row < num_rows; data_row++){
            vector<double> input_line;
            int base_idx = data_row * (input_layer_size - 1);
            for(int k=0; k<input_layer_size-1; k++) input_line.push_back(input[base_idx+k]);
            
            ForwardPropagation forward = forwardPropagation(input_line);
            
            vector<double> target_output;
            int out_base = data_row * output_layer_size;
            for(int k=0; k<output_layer_size; k++) target_output.push_back(output[out_base+k]);
            
            backPropagation(forward, input_line, target_output);
        }
        run();
    }
}

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

void Network::backPropagation(ForwardPropagation forward, vector<double> input_line, vector<double> output_line){

    input_line.push_back(1); // bias
    
    BackPropagation back(hidden_layer_size);

    for (int i = 0; i < output_layer_size; i++ ){
        back.delta_output_sum.push_back((output_line[i] - forward.output[i]) * sigmoidPrime(forward.sum_output_weigth[i]));
    }

    for (int i = 0; i < hidden_layer_size; i++ ){
        for (int j = 0; j < output_layer_size; j++ ){
            back.delta_input_sum[i] += back.delta_output_sum[j] * weight_output[i * output_layer_size + j];
        }
        back.delta_input_sum[i] *= sigmoidPrime(forward.sum_input_weight[i]);
    }

    for (unsigned int i = 0; i < hidden_layer_size; i++){
        for (unsigned int j = 0; j < output_layer_size; j++){
            weight_output[i * output_layer_size + j] += back.delta_output_sum[j] * forward.sum_input_weight_ativation[i] * learning_rate;
        }        
    }

    for (unsigned int i = 0; i < input_layer_size; i++){
        for (unsigned int j = 0; j < hidden_layer_size; j++){
            weight_input[i * hidden_layer_size + j] += back.delta_input_sum[j] * input_line[i] * learning_rate;
        }        
    }
}

void Network::hitRateCalculate(){
    hit_percent = (correct_output*100) / (input.size() / (input_layer_size - 1));
}

void Network::initializeWeight(){
    weight_input.resize(input_layer_size * hidden_layer_size);
    weight_output.resize(hidden_layer_size * output_layer_size);
    
    // Use thread id for seed to ensure diversity
    srand((unsigned int) time(0) ^ omp_get_thread_num());
    
    for(int i=0; i<weight_input.size(); i++) 
        weight_input[i] = ((double) rand() / (RAND_MAX));

    for(int i=0; i<weight_output.size(); i++) 
        weight_output[i] = ((double) rand() / (RAND_MAX));

    hit_percent = 0;
    correct_output = 0;
}

double Network::sigmoid(double z){
    return 1/(1+exp(-z));
}	

double Network::sigmoidPrime(double z){
    return exp(-z) / ( pow(1+exp(-z),2) );
}

void Network::setMaxEpoch(int m){ max_epoch = m; }
void Network::setDesiredPercent(int d){ desired_percent = d; }
void Network::setHiddenLayerSize(int h){ hidden_layer_size = h; }
void Network::setLearningRate(double l){ learning_rate = l; }
void Network::setErrorTolerance(double e){ error_tolerance = e; }

void Network::setInput(vector<vector<double>> i){
    input_layer_size = i[0].size() + 1; // +1 bias
    input.clear();
    for(const auto& row : i) {
        for(double val : row) input.push_back(val);
    }
}

void Network::setOutput(vector<vector<double>> o){
    output_layer_size = o[0].size();
    output.clear();
    for(const auto& row : o) {
        for(double val : row) output.push_back(val);
    }
}

void Network::autoTraining(int, double){}
void Network::trainingTemporal(){}

}