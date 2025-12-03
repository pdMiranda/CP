#include "../include/Network.hpp"

namespace Neural {

Network::Network(const std::vector<std::vector<double>>& input, const std::vector<std::vector<double>>& output) {
    n_rows = input.size();
    n_cols_in = input[0].size();
    n_cols_out = output[0].size();

    h_input = new double[n_rows * n_cols_in];
    h_output = new double[n_rows * n_cols_out];

    // Linearizar
    for(int i=0; i<n_rows; i++) {
        for(int j=0; j<n_cols_in; j++) h_input[i*n_cols_in + j] = input[i][j];
        for(int j=0; j<n_cols_out; j++) h_output[i*n_cols_out + j] = output[i][j];
    }
}

Network::~Network() {
    delete[] h_input;
    delete[] h_output;
}

}