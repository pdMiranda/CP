/*
 * Tempo Sequencial:                 0.046890 s
 * Tempo Paralelo codigo completo:   0.258214 s
 * Tempo Paralelo apenas operações:  0.022848 s 
*/

/*
  Como o Scan depende muito de transferência de dados entre Host e Device, 
  o tempo total do paralelo foi maior do que o sequencial devido o overhead de transferencia de memoria,
  por isso foi tambem calculado o tempo apenas das operações, so para ver o real impacto do paralelismo e do overhead
*/

/***
    Architecture:             x86_64
        CPU op-mode(s):         32-bit, 64-bit
        Address sizes:          39 bits physical, 48 bits virtual
        Byte Order:             Little Endian
        Model name:             11th Gen Intel(R) Core(TM) i7-11800H @ 2.30GHz
            CPU family:           6
            Model:                141
            Thread(s) per core:   2
            Core(s) per socket:   8
            Socket(s):            1
            Stepping:             1
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// Scan dentro dos blocos
__global__ void scan_cuda(double* a, double *s, int width) {
    int t = threadIdx.x;
    int b = blockIdx.x * blockDim.x;

    // Memória compartilhada para o scan do bloco
    __shared__ double p[1024];

    // Carrega dados da memória global para compartilhada
    if (b + t < width)
        p[t] = a[b + t];
    else
        p[t] = 0.0;

    __syncthreads();

    // Redução na memória compartilhada
    double x;
    for (int i = 1; i < blockDim.x; i *= 2) {
        if (t >= i)
            x = p[t] + p[t - i];
        
        __syncthreads();
        
        if (t >= i)
            p[t] = x;
        
        __syncthreads();
    }

    // Copia resultado de volta para memória global
    if (b + t < width)
        a[b + t] = p[t];

    // A última thread salva a soma total do bloco no vetor auxiliar 's'
    if (t == blockDim.x - 1)
        s[blockIdx.x] = p[t];
}

// Soma dos offsets
__global__ void add_cuda(double *a, double *s, int width) {
    int t = threadIdx.x;
    int b = blockIdx.x * blockDim.x;

    // Adiciona o acumulado dos blocos anteriores
    if (b + t < width && blockIdx.x > 0)
        a[b + t] += s[blockIdx.x];
}

int main()
{
  int width = 40000000;
  int size = width * sizeof(double);

  int block_size = 1024;
  int num_blocks = (width - 1) / block_size + 1;
  int s_size = (num_blocks * sizeof(double));  

  // Alocação no Host
  double *a = (double*) malloc (size);
  double *s = (double*) malloc (s_size);

  // Inicialização do vetor
  for(int i = 0; i < width; i++)
    a[i] = 1.0;

  double *d_a, *d_s;

  // Medição de tempo
  clock_t start = clock();

  // Alocar vetores "a" e "s" no device
  cudaMalloc((void **) &d_a, size);
  cudaMalloc((void **) &d_s, s_size);

  // Copiar vetor "a" para o device
  cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);

  // Definição do número de blocos e threads
  dim3 dimGrid(num_blocks, 1, 1);
  dim3 dimBlock(block_size, 1, 1);

  clock_t start_op = clock();
  // Chamada do kernel scan
  scan_cuda<<<dimGrid, dimBlock>>>(d_a, d_s, width);

  // Sincroniza para garantir que d_s esteja pronto
  cudaDeviceSynchronize();

  clock_t end_op = clock();

  // Copiar vetor "s" (somas parciais) para o host
  cudaMemcpy(s, d_s, s_size, cudaMemcpyDeviceToHost);

  // Scan no host
  // Transforma o vetor de somas em vetor de acumulados 
  double running_sum = 0.0;
  for (int i = 0; i < num_blocks; i++) {
      double temp = s[i];
      s[i] = running_sum;
      running_sum += temp;
  }

  // Copiar vetor "s" para o device
  cudaMemcpy(d_s, s, s_size, cudaMemcpyHostToDevice);

  // Chamada do kernel da soma
  add_cuda<<<dimGrid, dimBlock>>>(d_a, d_s, width);

  // Copiar o vetor "a" final para o host
  cudaMemcpy(a, d_a, size, cudaMemcpyDeviceToHost);

  // Medição de tempo
  clock_t end = clock();
  double time_spent = ((double)(end - start)) / CLOCKS_PER_SEC;
  double time_op = ((double)(end_op - start_op)) / CLOCKS_PER_SEC;

  printf("Tempo de execução: %f s\n", time_spent);
  printf("Tempo de operações: %f s\n", time_op);
  printf("\na[%d] = %f\n", width-1, a[width-1]);

  // Limpeza de memória
  cudaFree(d_a);
  cudaFree(d_s);
  free(a);
  free(s);

  return 0;
}