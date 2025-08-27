#include <stdio.h>
#include <omp.h>

int main()
{
    #pragma omp parallel for num_threads(2) schedule(static) // paraleliza o loop com 2 threads
    for(int i = 1; i <= 3; i++) 
    {
        int tid = omp_get_thread_num(); // lê o identificador da thread
        printf("[PRINT1] T%d = %d \n", tid, i);
        printf("[PRINT2] T%d = %d \n", tid, i);
    }
}