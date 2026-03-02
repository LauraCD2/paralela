#include <omp.h>
#include <stdio.h>

int main(void) {
  int threads = 0;

#pragma omp parallel reduction(+ : threads)
  threads += 1;

  printf("OpenMP enabled. Threads used: %d\n", threads);
  return 0;
}
