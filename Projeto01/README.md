https://github.com/alexandremstf/neural-network


# Neural Network C++
Este código foi desenvolvido usando STL, para criar redes neurais básicas em c + + com poucas linhas de código. É um projeto de iniciação científica da Universidade Federal de Ouro Preto, que utiliza como base de aprendizado o algoritmo de backpropagation. Com este código pessoas com pouco conhecimento sobre redes neurais podem as implementar. Existem duas maneiras de criar uma rede neural nesta biblioteca, manual ou automática, veja os exemplos abaixo:

Os arquivos que este código interpreta, são arquivos .txt onde cada linha representa os valores referentes a uma amostra, e os valores de cada amostra são separados por espaços. Deve-se atentar também ao fato de que os valores iniciais da linha devem ser os de entrada da rede e somente depois de todos os valores de entrada, insere-se os valores de saída correspondente (isto para etapa de treinamento) no arquivo. 

# Executar Código

Para compilar e executar o código, siga os passos abaixo. Existem três versões disponíveis: Sequencial, OpenMP e MPI. Escolha a versão desejada e siga as instruções correspondentes.

## Versão Sequencial

1. Compile o programa com o comando:

```bash
make all v=sequencial
```

2. Execute o programa com o comando:

```bash
./neuralnetwork
```

## Versão OpenMP

1. Compile o programa com o comando, especificando o número de threads desejado:

```bash
make all v=openmp t=<número_de_threads>
```

Substitua `<número_de_threads>` pelo número de threads que deseja utilizar.

2. Execute o programa com o comando:

```bash
./neuralnetwork
```
