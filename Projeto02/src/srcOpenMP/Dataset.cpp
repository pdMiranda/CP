#include "../include/Dataset.hpp"
#include <iomanip>
#include <omp.h> // Adicionado para OpenMP

namespace Neural {

Dataset::Dataset(){
}

void Dataset::saveOutputLog(){
}

void Dataset::printMatrix(vector<vector<double>> v){
    for (unsigned int i = 0; i < v.size(); i++){
        for (unsigned int j = 0; j < v[i].size(); j++){
            cout << fixed << setprecision(2) << v[i][j] << "\t";
        }
        cout << endl;
    }
    cout << endl;
}

void Dataset::printVector(vector<double> v){
    for (unsigned int i = 0; i < v.size(); i++){
        cout << fixed << setprecision(2) << v[i] << "\t";
    }
    cout << endl;
}

void Dataset::loadInputOutputData(int n_input, int n_output, string file){
    
    n_inputs = n_input;
    n_outputs = n_output;

    ifstream input;
        input.open(file);
    double data;

    if(!input.is_open()){
        cout << "FALHA ARQUIVO" << endl;
    } else {
        
        n_rows = int(count(istreambuf_iterator<char>(input), istreambuf_iterator<char>(), '\n')) + 1;
        
        input.clear();
        input.seekg(0, input.beg);

        input_data.resize(n_rows);
        output_data.resize(n_rows);

        for (int i = 0; i < n_rows; i++){
            for (int j = 0; j < n_inputs; j++){
                input >> data;
                input_data[i].push_back(data);
            }
            
            for (int j = 0; j < n_outputs; j++){
                input >> data;
                output_data[i].push_back(data);
            }
        }

        normalize(input_data);
    }

    input.close();
}

void Dataset::normalize(vector<vector<double>> v) {
    
    // O laço que itera sobre as colunas (i) é paralelizado.
    // Cada thread cuidará de uma ou mais colunas de forma independente.
    #pragma omp parallel for
    for (unsigned int i = 0; i < v[0].size(); i++){
        // As variáveis max e min são locais para cada thread, então não há conflito.
        double max = v[0][i];
        double min = v[0][i];
        
        for (unsigned int j = 0; j < v.size(); j++){
            if (max < v[j][i]) max = v[j][i];
            if (min > v[j][i]) min = v[j][i];
        }

        for (unsigned int j = 0; j < v.size(); j++){
             input_data[j][i] = (v[j][i] - min) / (max - min);
        }
    }

}

vector<vector<double>> Dataset::getInput(){
    return input_data;
}

vector<vector<double>> Dataset::getOutput(){
    return output_data;
}

}