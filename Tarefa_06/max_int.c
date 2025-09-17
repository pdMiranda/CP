#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define N 10
#define MAX 4
#define NUMBER 3

void main(int argc, char* argv[]) {
  int p, rank, maior_parcial, maior_final, numProcs;
  int buffer[N];
  MPI_Status status;

  MPI_Init(&argc, &argv) ;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &numProcs);

  if (rank == 0) {
    // preencher o buffer com N valores inteiros aleatórios
    srand(time(NULL));
    for (int i = 0; i < N; i++) {
      buffer[i] = rand() % 100; // Valores aleatórios de 0 a 99
    }
  } 

  // distribuir o vetor para todos os outros processos
  MPI_Bcast(buffer, N, MPI_INT, 0, MPI_COMM_WORLD);

  // processar o maior dos valores dentro do seu intervalo
  maior_parcial = -1; // Inicializa com um valor menor que qualquer possível no buffer
  for (int i = rank; i < N; i += numProcs) {
    if (buffer[i] > maior_parcial) {
      maior_parcial = buffer[i];
    }
  }

  // reduzir os maiores no maior, enviando o resultado para o processo com rank = 0
  MPI_Reduce(&maior_parcial, &maior_final, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);

  if (rank == 0) {
    // imprimir maior
    // imprimindo o vetor inteiro para verificacao.
    printf("Vetor: [ ");
    for(int i = 0; i < N; i++) {
        printf("%d ", buffer[i]);
    }
    printf("]\n");
    printf("Maior elemento encontrado: %d\n", maior_final);
  }

  MPI_Finalize();
}

// mpicc -o max_int max_int.c
// mpirun -np 4 ./max_int