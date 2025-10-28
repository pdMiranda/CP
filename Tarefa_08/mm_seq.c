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

// time 26.385786

//gcc -o mm_seq mm_seq.c -lrt -O3

#define _POSIX_C_SOURCE 199309L

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void mm(double* a, double* b, double* c, int width) 
{
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
