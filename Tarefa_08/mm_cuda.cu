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

// time 0.574730

//nvcc -o mm_cuda mm_cuda.cu -O3

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h> 

__global__ void mm_kernel(double* a, double* b, double* c, int width)
{
    // Índices 2D para linha (row) e coluna (col)
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Checagem de limites
    if (row < width && col < width) {

        // Granularidade por elemento, loop 'k' dentro do kernel
        double sum = 0;
        for (int k = 0; k < width; k++) {
            double x = a[row * width + k];
            double y = b[k * width + col];
            sum += x * y;
        }
        c[row * width + col] = sum;
    }
}


void mm(double* h_a, double* h_b, double* h_c, int width) 
{
  double *d_a, *d_b, *d_c; // Ponteiros para a memória da GPU
  int size = width * width * sizeof(double);

  cudaMalloc((void**)&d_a, size);
  cudaMalloc((void**)&d_b, size);
  cudaMalloc((void**)&d_c, size);

  cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

  // Configurar grid e blocos
  // blocos de 16x16 = 256 threads
  int THREADS_PER_BLOCK = 16; 
  dim3 dimBlock(THREADS_PER_BLOCK, THREADS_PER_BLOCK);
  
  int blocks = (width + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  dim3 dimGrid(blocks, blocks);

  // Lançar o kernel na GPU
  mm_kernel<<<dimGrid, dimBlock>>>(d_a, d_b, d_c, width);

  cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
}

int main()
{
  int width = 2000;
  double *a = (double*) malloc (width * width * sizeof(double));
  double *b = (double*) malloc (width * width * sizeof(double));
  double *c = (double*) malloc (width * width * sizeof(double));

  for(int i = 0; i < width; i++) {
    for(int j = 0; j < width; j++) {
      a[i*width+j] = i;
      b[i*width+j] = j;
      c[i*width+j] = 0;
    }
  }

  struct timespec start, end;
  clock_gettime(CLOCK_REALTIME, &start);
  double start_time = start.tv_sec + start.tv_nsec / 1e9;
  
  mm(a,b,c,width);
  
  cudaDeviceSynchronize(); 

  clock_gettime(CLOCK_REALTIME, &end);
  double end_time = end.tv_sec + end.tv_nsec / 1e9;

  printf("%f", end_time - start_time);

  // Limpeza
  free(a);
  free(b);
  free(c);

  //  for(int i = 0; i < width; i++) {
  //  for(int j = 0; j < width; j++) {
  //    printf("\n c[%d][%d] = %f",i,j,c[i*width+j]);
  //  }
  // }

}