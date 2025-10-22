# Neural Network C++
Este código-base foi desenvolvido usando STL, permitindo criar redes neurais básicas em C++ com poucas linhas de código. Ele é originário de um projeto de iniciação científica da Universidade Federal de Ouro Preto, que utiliza como fundamento de aprendizado o algoritmo de backpropagation. Com este código-base, pessoas com pouco conhecimento sobre redes neurais podem implementá-las. Existem duas maneiras de criar uma rede neural nesta biblioteca, manual ou automática, veja os exemplos abaixo:

Os arquivos que este código-base interpreta são arquivos .txt, onde cada linha representa os valores referentes a uma amostra, e os valores de cada amostra são separados por espaços. Deve-se atentar também ao fato de que os valores iniciais da linha devem ser os de entrada da rede e, somente depois de todos os valores de entrada, inserem-se os valores de saída correspondentes (isto para a etapa de treinamento) no arquivo.

https://github.com/alexandremstf/neural-network

# Sobre a Aplicação
O programa principal (`main.cpp`) é configurado para resolver um problema de classificação. Ele executa os seguintes passos:
  - Carrega o conjunto de dados "Iris" do arquivo `database/iris.txt`.
  - Configura os parâmetros da rede, como número máximo de épocas, taxa de acerto desejada e tolerância de erro.
  - Executa a função `autoTraining`. Esta função testa automaticamente várias combinações de hiperparâmetros (número de neurônios na camada oculta e taxa de aprendizado).
  - Ao final, o programa imprime no console a melhor combinação encontrada (aquela que atingiu a taxa de acerto no menor número de épocas) e o tempo total de execução.

## Dataset
O dataset Iris contém 150 amostras de três espécies de flores (Setosa, Virginica e Versicolor). Cada amostra é descrita por quatro características (comprimento e largura da sépala e da pétala), usadas para treinar uma rede neural a classificar corretamente a espécie da flor.

## Precepton com Backpropagation
A rede neural do projeto é um Perceptron com camadas de entrada, ocultas e de saída. Ela é treinada com o algoritmo backpropagation, que ajusta os pesos da rede para minimizar o erro entre a saída prevista e a desejada.


# Como Compilar e Executar

Este projeto utiliza um `makefile` para simplificar a compilação. O nome do executável gerado será `neuralnetwork`

- Compilar
  ```bash
  make all v=sequencial
  make all v=openmp t=<numero-de-threads>
  make all v=mpi t=<numero-de-threads> p=<numero-de-processos>
  ```

- Executar
  ```bash
    ./neuralnetwork
  ```
  
- Limpar Arquivos
    ```bash
        make clean
    ```


## Requisitos 

1. **Compilador C/C++**: Necessário para compilar o código em qualquer versão. Recomendado `gcc` ou `g++`.
2. **Make**: Utilizado para gerenciar a compilação do projeto.
3. **OpenMP**: Necessário para a versão paralela com OpenMP. Geralmente incluído no `gcc` (versão 4.2 ou superior).
4. **MPI (Message Passing Interface)**: Necessário para a versão distribuída com MPI. Recomendado `MPICH` ou `OpenMPI`.

# Grupo

- Andre Mendes
- Arthur Martinho
- Daniel Salgado
- Pedro Miranda