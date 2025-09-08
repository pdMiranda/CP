#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

// Compilar com: mpicc distribute_number.c -o distribute_number
// Executar com: mpirun -np 4 ./distribute_number (4 processos, 1 mestre e 3 escravos)

void main(int argc, char* argv[]) {
  int i, rank, val, numProcs;
  MPI_Status status;

  MPI_Init(&argc, &argv) ;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &numProcs);

  if (rank == 0) {
    val = 51;

    // enviar o valor para todos os outros processos (escravos)
    for (i = 1; i < numProcs; i++) {
      printf("Processo mestre %d enviando o valor %d para o processo %d\n", rank, val, i);
      MPI_Send(&val, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
    }

  } else {

    // receber o valor enviado pelo processo mestre (rank 0)
    MPI_Recv(&val, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
    printf("Processo escravo %d recebeu uma mensagem do processo %d com o valor %d\n", rank, 0, val);

  }

  MPI_Finalize();
}