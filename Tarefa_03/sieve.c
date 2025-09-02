/*
* Adapted from: http://w...content-available-to-author-only...s.org/sieve-of-eratosthenes
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

/***
    Execução Sequencial:
    5761455
        ____________________________________________________
        Executed in    1.27 secs    fish           external
            usr time    1.34 secs  175.00 micros    1.34 secs
            sys time    0.03 secs  210.00 micros    0.03 secs

    Execução Paralela:
    5761455
        ________________________________________________________
        Executed in  385.98 millis    fish           external
            usr time    5.69 secs    204.00 micros    5.69 secs
            sys time    0.03 secs    217.00 micros    0.03 secs
*/


#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>

#include <omp.h>
 
int sieveOfEratosthenes(int n)
{
   // Create a boolean array "prime[0..n]" and initialize
   // all entries it as true. A value in prime[i] will
   // finally be false if i is Not a prime, else true. 
   int primes = 0; 
   bool *prime = (bool*) malloc((n+1)*sizeof(bool));
   int sqrt_n = sqrt(n);

   memset(prime, true, (n+1)*sizeof(bool));

   #pragma omp parallel for schedule(dynamic)
   for (int p = 2; p <= sqrt_n; p++)
   {
       // If prime[p] is not changed, then it is a prime 
       if (prime[p] == true)
       {
           // Update all multiples of p 
           for (int i = p * 2; i <= n; i += p)
           {
               prime[i] = false;
           }
       }
   }

   #pragma omp parallel for reduction(+:primes)
   // count prime numbers
   for (int p = 2; p <= n; p++)
   {
       if (prime[p])
           primes++;
   }

   free(prime);
   return primes;
}
 
int main()
{
   int n = 100000000;
   printf("%d\n",sieveOfEratosthenes(n));
   return 0;
}



