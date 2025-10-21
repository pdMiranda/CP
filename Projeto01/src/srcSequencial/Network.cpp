#include "../include/Network.hpp"

namespace Neural{

Network::Network(){
}

Network::Network(vector<vector<double>> user_input, vector<vector<double>> user_output){

    setInput(user_input);
    setOutput(user_output);
    output_layer_size = 3;

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

void Network::run(){

    //#pragma omp parallel for num_threads(NUM_THREADS)
    for (unsigned int data_row = 0; data_row < input.size(); data_row++){
        ForwardPropagation forward = forwardPropagation(input[data_row]);
        hitRateCount(forward.output, data_row);            
    }
    hitRateCalculate();    
}

void Network::trainingClassification(){

    for (epoch = 0; epoch < max_epoch && hit_percent < desired_percent; epoch++) {
        for (unsigned int data_row = 0; data_row < input.size(); data_row++){
            ForwardPropagation forward = forwardPropagation(input[data_row]);
            backPropagation(forward, input[data_row], output[data_row]);
        }
        run();
    }

    cout << "Hidden Layer Size: " << hidden_layer_size 
        << "\tLearning Rate: " << learning_rate 
        << "\tHit Percent: " << hit_percent << "%" 
        << "\tEpoch: " << epoch << endl;
}

void Network::autoTraining(int hidden_layer_limit, double learning_rate_increase){

    for (hidden_layer_size = 3; hidden_layer_size <= hidden_layer_limit; hidden_layer_size++){
        for (learning_rate = learning_rate_increase; learning_rate <= 1; learning_rate = learning_rate + learning_rate_increase){
            initializeWeight();
            trainingClassification();
            if (epoch < best_network.epoch){
                best_network.epoch = epoch;
                best_network.learning_rate = learning_rate;
                best_network.hidden_layer = hidden_layer_size;
                best_network.weight_input = weight_input;
	            best_network.weight_output = weight_output;
            }
        }
    }

    cout << "Best Network --> Hidden Layer Size: " << best_network.hidden_layer 
        << "\tLearning Rate: " << best_network.learning_rate 
        << "\tEpoch: " << best_network.epoch << endl;
    
    epoch = best_network.epoch;
    learning_rate = best_network.learning_rate;
    hidden_layer_size = best_network.hidden_layer;
    weight_input = best_network.weight_input;
    weight_output = best_network.weight_output;
}

Network::ForwardPropagation Network::forwardPropagation(vector<double> input_line){

    input_line.push_back(1); // bias

    ForwardPropagation forward(hidden_layer_size, output_layer_size);

    // somatório dos produtos entre, entrada e peso das entradas em cada neurônio da camada oculta
    for (int i = 0; i < hidden_layer_size; i++ ){
        for (int j = 0; j < input_layer_size; j++ ){
            forward.sum_input_weight[i] += input_line[j] * weight_input[j][i];
        }
    }

    // aplica função de ativação, em cada somatório encontrado, ou em cada neurônio da camada oculta  (sigmoid)
    for (int i = 0; i < hidden_layer_size; i++ ){
        forward.sum_input_weight_ativation.push_back(sigmoid(forward.sum_input_weight[i]));
    }

    // somatório dos produtos entre, o somatório dos neurônios na camada oculta e o peso das saídas
    for (int i = 0; i < output_layer_size; i++ ){
        for (int j = 0; j < hidden_layer_size; j++ ){
            forward.sum_output_weigth[i] += forward.sum_input_weight_ativation[j] * weight_output[j][i];
        }
    }

    // aplica função de ativação em cada somatório encontrado, ou em cada nerônio da camada de saída (sigmoidPrime), saída da rede neural
    for (int i = 0; i < output_layer_size; i++ ){
        forward.output.push_back(sigmoid(forward.sum_output_weigth[i]));
    }

    return forward;
}

void Network::backPropagation(ForwardPropagation forward, vector<double> input_line, vector<double> output_line){

    input_line.push_back(1); // bias
    
    BackPropagation back(hidden_layer_size);

    // erro entre a saída esperada e a calculada, multiplicado pela taxa de mudança da função de ativação no somatório de saída (derivada)
    for (int i = 0; i < output_layer_size; i++ ){
        back.delta_output_sum.push_back((output_line[i] - forward.output[i]) * sigmoidPrime(forward.sum_output_weigth[i]));
    }

    // erro da saída multiplicado pelos pesos de saída, aplicando a taxa de mudança da função de ativação no somatório da camada oculta (derivada)
    for (int i = 0; i < hidden_layer_size; i++ ){
        for (int j = 0; j < output_layer_size; j++ ){
            back.delta_input_sum[i] += back.delta_output_sum[j] * weight_output[i][j];
        }
        back.delta_input_sum[i] *= sigmoidPrime(forward.sum_input_weight[i]);
    }

    // corrigindo os valores dos pesos de saída
    for (unsigned int i = 0; i < weight_output.size(); i++){
        for (unsigned int j = 0; j < weight_output[i].size(); j++){
            weight_output[i][j] += back.delta_output_sum[j] * forward.sum_input_weight_ativation[i] * learning_rate;
        }        
    }

    // corrigindo os valores dos pesos de entrada
    for (unsigned int i = 0; i < weight_input.size(); i++){
        for (unsigned int j = 0; j < weight_input[i].size(); j++){
            weight_input[i][j] += back.delta_input_sum[j] * input_line[i] * learning_rate;
        }        
    }
}

void Network::hitRateCount(vector<double> neural_output, unsigned int data_row){

    for (int i = 0; i < output_layer_size; i++ ){
        if (abs(neural_output[i] - output[data_row][i]) < error_tolerance)
            correct_output++;
    }
}

void Network::hitRateCalculate(){

    hit_percent = (correct_output*100) / (output.size() * output_layer_size);
    correct_output = 0;
}

void Network::initializeWeight(){

    weight_input.resize(input_layer_size);
    weight_output.resize(hidden_layer_size);
    
    srand((unsigned int) time(0));
    
    for (unsigned int i = 0; i < weight_input.size(); i++ ){
        weight_input[i].clear();
        for ( int j = 0; j < hidden_layer_size; j++ ){
            weight_input[i].push_back(((double) rand() / (RAND_MAX)));
        }
    }

    for (unsigned int i = 0; i < weight_output.size(); i++ ){
        weight_output[i].clear();        
        for ( int j = 0; j < output_layer_size; j++ ){
            weight_output[i].push_back(((double) rand() / (RAND_MAX)));
        }
    }

    hit_percent = 0;
    correct_output = 0;
}

double Network::sigmoid(double z){
    return 1/(1+exp(-z));
}	

double Network::sigmoidPrime(double z){
    return exp(-z) / ( pow(1+exp(-z),2) );
}

void Network::setMaxEpoch(int m){
    max_epoch = m;
}

void Network::setDesiredPercent(int d){
    desired_percent = d;
}

void Network::setHiddenLayerSize(int h){
    hidden_layer_size = h;
}

void Network::setLearningRate(double l){
    learning_rate = l;
}

void Network::setErrorTolerance(double e){
    error_tolerance = e;
}

void Network::setInput(vector<vector<double>> i){
    input = i;
    input_layer_size = i[0].size() + 1; // +1 bias
}

void Network::setOutput(vector<vector<double>> o){
    output = o;
	output_layer_size = o[0].size();    
}

}