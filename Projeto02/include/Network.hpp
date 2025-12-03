#ifndef NETWORK_HPP_
#define NETWORK_HPP_

#include <iostream>
#include <vector>
#include <cmath>
#include <cstring>

namespace Neural {

struct NetworkConfig {
    int input_size;
    int output_size;
    int hidden_size;
    int max_epoch;
    double learning_rate;
    double error_tolerance;
    int desired_percent;
};

class Network {
private:
    double* h_input;
    double* h_output;
    int n_rows;
    int n_cols_in;
    int n_cols_out;

public:
    Network(const std::vector<std::vector<double>>& input, const std::vector<std::vector<double>>& output);
    ~Network();

    void trainOpenMP(int num_models, int num_teams, int num_threads);
    void trainCUDA(int num_models, int num_blocks, int num_threads);

    static double sigmoid(double z);
    static double sigmoidPrime(double z);
};

}

#endif