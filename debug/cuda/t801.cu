// this demonstrates that failure to allocate memory doesn't corrupt the CUDA context
//
// If the CUDA context has not been corrupted then the state can be reset to cudaSuccess by calling cudaGetLastError().
//
// Ordinary cudaMalloc operations return no error. A cudaMalloc operation that runs out of memory will return error 2. A subsequent call to cudaGetLastError() will
// return no error, because the error 2 does not corrupt the cuda context, and is therefore not a "sticky" error. Subsequent operations after that also return no
// error
// from: https://stackoverflow.com/a/30911340/9201239

#include <stdio.h>

#define DSIZE 20000000000ULL

int main(){

  int *d1, *d2, *d3;
  cudaMalloc(&d1, 4);
  printf("err1 :  %d\n", (int)cudaGetLastError());
  cudaMalloc(&d2, DSIZE);
  printf("err2a:  %d\n", (int)cudaGetLastError());
  printf("err2b:  %d\n", (int)cudaGetLastError());
  cudaFree(d1);
  printf("err3 :  %d\n", (int)cudaGetLastError());
  cudaMalloc(&d3, 4);
  printf("err4 :  %d\n", (int)cudaGetLastError());
}
