#include "../include/Network.hpp"

#ifdef NUM_THREADS
#define THREADS NUM_THREADS
#else
#define THREADS 1
#endif

namespace Neural
{

    Network::Network()
    {
        omp_set_num_threads(THREADS);
    }

    Network::Network(double *user_input, double *user_output, int input_size, int output_size)
    {
        setInput(user_input, input_size, output_size);
        setOutput(user_output, input_size, output_size);
        output_layer_size = 3;
        omp_set_num_threads(THREADS);
    }

    void Network::setParameter(int user_max_epoch, int user_desired_percent, double user_error_tolerance, double user_learning_rate, int user_hidden_layer_size)
    {

        setMaxEpoch(user_max_epoch);
        setLearningRate(user_learning_rate);
        setErrorTolerance(user_error_tolerance);
        setDesiredPercent(user_desired_percent);
        setHiddenLayerSize(user_hidden_layer_size);
        best_network.epoch = max_epoch;

        initializeWeight();
    }

    void Network::run()
    {
        for (int data_row = 0; data_row < output_rows; data_row++)
        {
            int input_start = data_row * input_layer_size;
            ForwardPropagation forward = forwardPropagation(&input[input_start]);
            hitRateCount(forward.output, data_row);
        }
        hitRateCalculate();
    }

    void Network::trainingClassification()
    {
        for (epoch = 0; epoch < max_epoch && hit_percent < desired_percent; epoch++)
        {
            for (int data_row = 0; data_row < output_rows; data_row++)
            {
                int input_start = data_row * input_layer_size;
                int output_start = data_row * output_layer_size;

                ForwardPropagation forward = forwardPropagation(&input[input_start]);
                backPropagation(forward, &input[input_start], &output[output_start]);
            }
            run();
        }

        printf("Hidden Layer Size: %d\tLearning Rate: %f\tHit Percent: %d%%\tEpoch: %d\n",
               hidden_layer_size, learning_rate, hit_percent, epoch);
    }

    void Network::autoTraining(int hidden_layer_limit, double learning_rate_increase)
    {
        network global_best_network = best_network;
        Network *this_ptr = this;

        // Calcula o número total de iterações
        int num_hidden_layers = hidden_layer_limit - 2; // de 3 até hidden_layer_limit
        int num_learning_rates = (int)(1.0 / learning_rate_increase);
        int total_iterations = num_hidden_layers * num_learning_rates;

        // Aloca arrays para armazenar resultados temporários
        int *epochs = new int[total_iterations];
        double *learning_rates = new double[total_iterations];
        int *hidden_layers = new int[total_iterations];
        double *temp_weights_input = new double[input_weight_size * total_iterations];
        double *temp_weights_output = new double[output_weight_size * total_iterations];

        {

            for (int i = 3; i <= hidden_layer_limit; i++)
            {
                for (int j = 0; j < num_learning_rates; j++)
                {
                    double current_learning_rate = (j + 1) * learning_rate_increase;
                    int idx = (i - 3) * num_learning_rates + j;

                    Network local_network = *this_ptr;
                    local_network.hidden_layer_size = i;
                    local_network.learning_rate = current_learning_rate;
                    local_network.initializeWeight();
                    local_network.trainingClassification();

                    // Armazena resultados nos arrays temporários
                    epochs[idx] = local_network.epoch;
                    learning_rates[idx] = current_learning_rate;
                    hidden_layers[idx] = i;

                    // Copia os pesos para os arrays temporários
                    int weight_input_offset = idx * input_weight_size;
                    int weight_output_offset = idx * output_weight_size;

                    for (int k = 0; k < input_weight_size; k++)
                    {
                        temp_weights_input[weight_input_offset + k] = local_network.weight_input[k];
                    }

                    for (int k = 0; k < output_weight_size; k++)
                    {
                        temp_weights_output[weight_output_offset + k] = local_network.weight_output[k];
                    }
                }
            }
        }

        // Encontra o melhor resultado
        int best_idx = 0;
        for (int i = 1; i < total_iterations; i++)
        {
            if (epochs[i] < epochs[best_idx])
            {
                best_idx = i;
            }
        }

        // Atualiza a rede com os melhores parâmetros encontrados
        epoch = epochs[best_idx];
        learning_rate = learning_rates[best_idx];
        hidden_layer_size = hidden_layers[best_idx];

        // Copia os melhores pesos
        int best_weight_input_offset = best_idx * input_weight_size;
        int best_weight_output_offset = best_idx * output_weight_size;

        memcpy(weight_input, &temp_weights_input[best_weight_input_offset],
               input_weight_size * sizeof(double));
        memcpy(weight_output, &temp_weights_output[best_weight_output_offset],
               output_weight_size * sizeof(double));

        std::cout << "Best Network --> Hidden Layer Size: " << hidden_layer_size
                  << "\tLearning Rate: " << learning_rate
                  << "\tEpoch: " << epoch << std::endl;

        // Libera memória
        delete[] epochs;
        delete[] learning_rates;
        delete[] hidden_layers;
        delete[] temp_weights_input;
        delete[] temp_weights_output;
    }

    Network::ForwardPropagation Network::forwardPropagation(double *input_line)
    {
        // Cria uma nova instância de ForwardPropagation
        ForwardPropagation forward(hidden_layer_size, output_layer_size);

        // Calcula o somatório dos produtos entre entrada e pesos
        for (int i = 0; i < hidden_layer_size; i++)
        {
            forward.sum_input_weight[i] = 0;
            for (int j = 0; j < input_layer_size; j++)
            {
                forward.sum_input_weight[i] += input_line[j] * weight_input[j * hidden_layer_size + i];
            }
        }

        // Aplica função de ativação
        for (int i = 0; i < hidden_layer_size; i++)
        {
            forward.sum_input_weight_activation[i] = sigmoid(forward.sum_input_weight[i]);
        }

        // Calcula saídas
        for (int i = 0; i < output_layer_size; i++)
        {
            forward.sum_output_weight[i] = 0;
            for (int j = 0; j < hidden_layer_size; j++)
            {
                forward.sum_output_weight[i] += forward.sum_input_weight_activation[j] *
                                                weight_output[j * output_layer_size + i];
            }
            forward.output[i] = sigmoid(forward.sum_output_weight[i]);
        }

        return std::move(forward);
    }

    void Network::backPropagation(ForwardPropagation &forward, double *input_line, double *output_line)
    {
        double *delta_output = new double[output_layer_size];
        double *delta_input = new double[hidden_layer_size];

        // Calcula erro de saída
        for (int i = 0; i < output_layer_size; i++)
        {
            delta_output[i] = (output_line[i] - forward.output[i]) *
                              sigmoidPrime(forward.sum_output_weight[i]);
        }

        // Calcula erro da camada oculta
        for (int i = 0; i < hidden_layer_size; i++)
        {
            delta_input[i] = 0;
            for (int j = 0; j < output_layer_size; j++)
            {
                delta_input[i] += delta_output[j] * weight_output[i * output_layer_size + j];
            }
            delta_input[i] *= sigmoidPrime(forward.sum_input_weight[i]);
        }

        // Atualiza pesos
        for (int i = 0; i < hidden_layer_size; i++)
        {
            for (int j = 0; j < output_layer_size; j++)
            {
                weight_output[i * output_layer_size + j] += learning_rate *
                                                            delta_output[j] * forward.sum_input_weight_activation[i];
            }
        }

        for (int i = 0; i < input_layer_size; i++)
        {
            for (int j = 0; j < hidden_layer_size; j++)
            {
                weight_input[i * hidden_layer_size + j] += learning_rate *
                                                           delta_input[j] * input_line[i];
            }
        }

        delete[] delta_output;
        delete[] delta_input;
    }

    void Network::hitRateCount(double *neural_output, unsigned int data_row)
    {
        for (int i = 0; i < output_layer_size; i++)
        {
            // Acessa o output linearizado usando data_row * output_layer_size + i
            if (abs(neural_output[i] - output[data_row * output_layer_size + i]) < error_tolerance)
                correct_output++;
        }
    }

    void Network::hitRateCalculate()
    {

        hit_percent = (correct_output * 100) / (output_rows * output_layer_size);
        correct_output = 0;
    }

    void Network::initializeWeight()
    {
        input_weight_size = input_layer_size * hidden_layer_size;
        output_weight_size = hidden_layer_size * output_layer_size;

        // Aloca memória para os arrays
        weight_input = new double[input_weight_size];
        weight_output = new double[output_weight_size];

        srand((unsigned int)time(0));

        // Inicializa os pesos de entrada
        for (int i = 0; i < input_weight_size; i++)
        {
            weight_input[i] = ((double)rand() / (RAND_MAX));
        }

        // Inicializa os pesos de saída
        for (int i = 0; i < output_weight_size; i++)
        {
            weight_output[i] = ((double)rand() / (RAND_MAX));
        }

        hit_percent = 0;
        correct_output = 0;
    }

    double Network::sigmoid(double z)
    {
        return 1 / (1 + exp(-z));
    }

    double Network::sigmoidPrime(double z)
    {
        return exp(-z) / (pow(1 + exp(-z), 2));
    }

    void Network::setMaxEpoch(int m)
    {
        max_epoch = m;
    }

    void Network::setDesiredPercent(int d)
    {
        desired_percent = d;
    }

    void Network::setHiddenLayerSize(int h)
    {
        hidden_layer_size = h;
    }

    void Network::setLearningRate(double l)
    {
        learning_rate = l;
    }

    void Network::setErrorTolerance(double e)
    {
        error_tolerance = e;
    }

    void Network::setInput(double *i, int rows, int cols)
    {
        input = new double[rows * cols];
        memcpy(input, i, rows * cols * sizeof(double));
        input_layer_size = cols + 1; // +1 para bias
    }

    void Network::setOutput(double *o, int rows, int cols)
    {
        output_rows = rows; // Armazena o número de linhas
        output = new double[rows * cols];
        for (int j = 0; j < rows * cols; j++)
        {
            output[j] = o[j];
        }
        output_layer_size = cols;
    }

    Network::~Network()
    {
        delete[] input;
        delete[] output;
        delete[] weight_input;
        delete[] weight_output;
    }

}