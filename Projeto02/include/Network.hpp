#ifndef NETWORK_HPP_
#define NETWORK_HPP_

#include <iostream>
#include <vector>
#include <cmath>
#include <omp.h>
#include <cstring>

namespace Neural
{

    class Network
    {

#pragma omp declare target
        struct ForwardPropagation
        {
            static const int MAX_SIZE = 1000;
            double sum_input_weight[MAX_SIZE];
            double sum_output_weight[MAX_SIZE];
            double sum_input_weight_activation[MAX_SIZE];
            double output[MAX_SIZE];
            int input_size;
            int output_size;

            ForwardPropagation() : input_size(0), output_size(0)
            {
                for (int i = 0; i < MAX_SIZE; i++)
                {
                    sum_input_weight[i] = 0.0;
                    sum_output_weight[i] = 0.0;
                    sum_input_weight_activation[i] = 0.0;
                    output[i] = 0.0;
                }
            }

            ForwardPropagation(int size_input, int size_output) : ForwardPropagation()
            {
                input_size = size_input;
                output_size = size_output;
            }
        };
#pragma omp end declare target

        struct network
        {
            int epoch;
            int hidden_layer;
            double learning_rate;
            double *weight_input;
            double *weight_output;
        };

    private:
        int input_layer_size;
        int output_layer_size;
        int hidden_layer_size;

        double *input;         // Dados de entrada linearizados
        double *output;        // Dados de saída linearizados
        double *weight_input;  // Pesos de entrada linearizados
        double *weight_output; // Pesos de saída linearizados

        network best_network;

        int output_rows;

        int epoch;
        int max_epoch;

        int correct_output;
        int hit_percent;

        double desired_percent;
        double learning_rate;
        double error_tolerance;

        int input_weight_size;
        int output_weight_size;

    public:
#pragma omp declare target
        Network();
        void initializeWeight();
        void run();
        void forwardPropagation(double *input_line, ForwardPropagation &forward);
        void backPropagation(ForwardPropagation &, double *, double *);

        double sigmoid(double);
        double sigmoidPrime(double);
        void hitRateCount(double *, unsigned int);
        void hitRateCalculate();
        void trainOneEpoch();
#pragma omp end declare target
        ~Network();
        void trainingClassification();
        void autoTraining(int, double);
        Network(double *, double *, int, int, int, int);

        void setInput(double *, int, int);
        void setOutput(double *, int, int);
        void setMaxEpoch(int);
        void setDesiredPercent(int);
        void setHiddenLayerSize(int);
        void setLearningRate(double);
        void setErrorTolerance(double);
        void setParameter(int, int, double, double = 1, int = 1);
    };

}

#endif
