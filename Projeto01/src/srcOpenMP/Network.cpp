#include "../include/Network.hpp"
#include <omp.h> // Adicionado para OpenMP

#ifndef NUM_THREADS
#define NUM_THREADS omp_get_max_threads()
#endif

namespace Neural{

Network::Network(){
    // Define o número de threads no início
    omp_set_num_threads(NUM_THREADS);
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
    // A variável 'correct_output' será reiniciada para esta chamada
    correct_output = 0;
    
    // A cláusula 'reduction(+:correct_output)' cria uma cópia local de correct_output
    // para cada thread. No final, todas as cópias são somadas na variável original.
    #pragma omp parallel for reduction(+:correct_output)
    for (unsigned int data_row = 0; data_row < input.size(); data_row++){
        ForwardPropagation forward = forwardPropagation(input[data_row]);
        
        // A lógica de hitRateCount foi movida para cá para simplificar a paralelização
        for (int i = 0; i < output_layer_size; i++ ){
            if (abs(forward.output[i] - output[data_row][i]) < error_tolerance)
                correct_output++;
        }          
    }
    hitRateCalculate();    
}

void Network::trainingClassification(){
    // Nota: Esta função em si não foi paralelizada para manter a lógica sequencial das épocas.
    // A paralelização ocorre no nível superior (autoTraining), que é muito mais eficiente.
    for (epoch = 0; epoch < max_epoch && hit_percent < desired_percent; epoch++) {
        for (unsigned int data_row = 0; data_row < input.size(); data_row++){
            ForwardPropagation forward = forwardPropagation(input[data_row]);
            backPropagation(forward, input[data_row], output[data_row]);
        }
        run(); // A função run() chamada aqui agora é a versão paralela
    }

    // A impressão foi movida para dentro do autoTraining para evitar poluição no console.
}

void Network::autoTraining(int hidden_layer_limit, double learning_rate_increase){
    
    cout << "Iniciando auto-treinamento paralelo em " << omp_get_max_threads() << " threads..." << endl;
    best_network.epoch = max_epoch + 1; // Garante que qualquer resultado seja melhor no início

    // Calculamos o número de passos que o laço de learning_rate precisa dar.
    // Ex: se o incremento é 0.1, teremos 10 passos (0.1, 0.2, ..., 1.0)
    int learning_rate_steps = static_cast<int>(1.0 / learning_rate_increase);

    // Agora, os dois laços usam variáveis do tipo 'int', o que é válido para o OpenMP.
    #pragma omp parallel for collapse(2)
    for (int h_size = 3; h_size <= hidden_layer_limit; h_size++){
        for (int i = 1; i <= learning_rate_steps; i++){
            
            // Calculamos o 'l_rate' real a partir do passo do laço de inteiros 'i'
            double l_rate = i * learning_rate_increase;
            
            // 1. CRIA UMA CÓPIA LOCAL DA REDE PARA ESTA THREAD
            Network thread_net = *this;

            // 2. CONFIGURA E TREINA A CÓPIA LOCAL
            thread_net.setHiddenLayerSize(h_size);
            thread_net.setLearningRate(l_rate);
            
            thread_net.initializeWeight(); 
            
            thread_net.trainingClassification();

            // 3. COMPARA O RESULTADO DE FORMA SEGURA
            #pragma omp critical
            {
                // Esta linha imprime o resultado de CADA combinação de hiperparâmetros.
                cout << "Hidden Layer Size: " << thread_net.hidden_layer_size 
                     << "\tLearning Rate: " << thread_net.learning_rate 
                     << "\tHit Percent: " << thread_net.hit_percent << "%" 
                     << "\tEpoch: " << thread_net.epoch << endl;

                // Atualiza a melhor rede encontrada (lógica anterior)
                if (thread_net.epoch < best_network.epoch){
                    best_network.epoch = thread_net.epoch;
                    best_network.learning_rate = thread_net.learning_rate;
                    best_network.hidden_layer = thread_net.hidden_layer_size;
                    best_network.weight_input = thread_net.weight_input;
	                best_network.weight_output = thread_net.weight_output;
                }
            }
        }
    }

    cout << "\n-----------------------------------------------------" << endl;
    cout << "Treinamento finalizado." << endl;
    cout << "Melhor Rede Encontrada --> Hidden Layer: " << best_network.hidden_layer 
        << "\tLearning Rate: " << best_network.learning_rate 
        << "\tEpoch: " << best_network.epoch << endl;
    cout << "-----------------------------------------------------" << endl;
    
    // Carrega os parâmetros da melhor rede para o objeto principal
    epoch = best_network.epoch;
    learning_rate = best_network.learning_rate;
    hidden_layer_size = best_network.hidden_layer;
    weight_input = best_network.weight_input;
    weight_output = best_network.weight_output;
}

// ... As funções forwardPropagation e backPropagation permanecem inalteradas ...
Network::ForwardPropagation Network::forwardPropagation(vector<double> input_line){

    input_line.push_back(1); // bias

    ForwardPropagation forward(hidden_layer_size, output_layer_size);

    for (int i = 0; i < hidden_layer_size; i++ ){
        for (int j = 0; j < input_layer_size; j++ ){
            forward.sum_input_weight[i] += input_line[j] * weight_input[j][i];
        }
    }

    for (int i = 0; i < hidden_layer_size; i++ ){
        forward.sum_input_weight_ativation.push_back(sigmoid(forward.sum_input_weight[i]));
    }

    for (int i = 0; i < output_layer_size; i++ ){
        for (int j = 0; j < hidden_layer_size; j++ ){
            forward.sum_output_weigth[i] += forward.sum_input_weight_ativation[j] * weight_output[j][i];
        }
    }

    for (int i = 0; i < output_layer_size; i++ ){
        forward.output.push_back(sigmoid(forward.sum_output_weigth[i]));
    }

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
            back.delta_input_sum[i] += back.delta_output_sum[j] * weight_output[i][j];
        }
        back.delta_input_sum[i] *= sigmoidPrime(forward.sum_input_weight[i]);
    }

    for (unsigned int i = 0; i < weight_output.size(); i++){
        for (unsigned int j = 0; j < weight_output[i].size(); j++){
            weight_output[i][j] += back.delta_output_sum[j] * forward.sum_input_weight_ativation[i] * learning_rate;
        }        
    }

    for (unsigned int i = 0; i < weight_input.size(); i++){
        for (unsigned int j = 0; j < weight_input[i].size(); j++){
            weight_input[i][j] += back.delta_input_sum[j] * input_line[i] * learning_rate;
        }        
    }
}


// A função hitRateCount não é mais necessária, pois sua lógica foi incorporada em run()
/*
void Network::hitRateCount(vector<double> neural_output, unsigned int data_row){
    for (int i = 0; i < output_layer_size; i++ ){
        if (abs(neural_output[i] - output[data_row][i]) < error_tolerance)
            correct_output++;
    }
}
*/

void Network::hitRateCalculate(){

    hit_percent = (correct_output*100.0) / (output.size() * output_layer_size);
    // correct_output é zerado no início da função run() agora.
}

void Network::initializeWeight(){

    weight_input.resize(input_layer_size);
    weight_output.resize(hidden_layer_size);
    
    // Adicionamos o ID da thread à semente para garantir valores aleatórios diferentes
    // caso esta função seja chamada de dentro de um laço paralelo.
    srand((unsigned int) time(0) ^ omp_get_thread_num());
    
    for (unsigned int i = 0; i < weight_input.size(); i++ ){
        weight_input[i].assign(hidden_layer_size, 0.0); // .assign é mais eficiente que clear+push_back
        for ( int j = 0; j < hidden_layer_size; j++ ){
            weight_input[i][j] = ((double) rand() / (RAND_MAX));
        }
    }

    for (unsigned int i = 0; i < weight_output.size(); i++ ){
        weight_output[i].assign(output_layer_size, 0.0);        
        for ( int j = 0; j < output_layer_size; j++ ){
            weight_output[i][j] = ((double) rand() / (RAND_MAX));
        }
    }

    hit_percent = 0;
    correct_output = 0;
}

// ... O resto do arquivo (sigmoid, sigmoidPrime, setters, getters) permanece o mesmo ...
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