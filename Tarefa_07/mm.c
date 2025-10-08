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

/*
gcc -O3 mm_sequencial.c -o sequencial
  Tempo sequencial: 26.815124 segundos

gcc -O3 -fopenmp mm_multicore.c -o multicore
  Tempo paralelo (multicore): 5.093410 segundos

gcc -O3 -fopenmp mm_gpu_distribute.c -o gpu_distribute
  Tempo GPU (distribute): 25.415343 segundos

gcc -O3 -fopenmp mm_gpu_dist_par_for.c -o gpu_dist_par_for
  Tempo GPU (distribute parallel for): 5.386845 segundos

gcc -O3 -fopenmp mm_gpu_dist_par_for_simd.c -o gpu_dist_par_for_simd
  Tempo GPU (distribute parallel for simd): 5.430972 segundos
*/



#include <stdio.h>
#include <stdlib.h>

void mm(double* a, double* b, double* c, int width) 
{
  //#pragma omp parallel for collapse(2)
  //#pragma omp target teams distribute map(to:a[0:width*width], b[0:width*width]) map(from:c[0:width*width])
  //#pragma omp target teams distribute parallel for map(to:a[0:width*width], b[0:width*width]) map(from:c[0:width*width])
  //#pragma omp target teams distribute parallel for simd collapse(2) map(to:a[0:width*width], b[0:width*width]) map(from:c[0:width*width])
  for (int i = 0; i < width; i++) {
    for (int j = 0; j < width; j++) {
      double sum = 0;
      for (int k = 0; k < width; k++) {
	double x = a[i * width + k];
	double y = b[k * width + j];
	sum += x * y;
      }
      c[i * width + j] = sum;
    }
  }
}

int main()
{
  int width = 2000;
  double *a = (double*) malloc (width * width * sizeof(double));
  double *b = (double*) malloc (width * width * sizeof(double));
  double *c = (double*) malloc (width * width * sizeof(double));

  //#pragma omp parallel for
  for(int i = 0; i < width; i++) {
    for(int j = 0; j < width; j++) {
      a[i*width+j] = i;
      b[i*width+j] = j;
      c[i*width+j] = 0;
    }
  }

  double start_time = omp_get_wtime();
  mm(a,b,c,width);
  double end_time = omp_get_wtime();
  printf("%f", end_time - start_time);

  free(a);
  free(b);
  free(c);

  return 0;

  //  for(int i = 0; i < width; i++) {
  //  for(int j = 0; j < width; j++) {
  //    printf("\n c[%d][%d] = %f",i,j,c[i*width+j]);
  //  }
  // }

}
