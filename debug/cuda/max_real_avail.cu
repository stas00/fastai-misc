/* calculate how much memory can be really allocated (which is not the same as free)
   https://stackoverflow.com/a/8923966/9201239
*/

#include <stdio.h>
#include <cuda.h>
#include <unistd.h>

const size_t Mb = 1<<20; // Assuming a 1Mb page size here

int main() {

    size_t total;
    size_t avail;
    cudaError_t cuda_status = cudaMemGetInfo(&avail, &total);
    if ( cudaSuccess != cuda_status ) {
      printf("Error: cudaMemGetInfo fails, %s \n", cudaGetErrorString(cuda_status) );
      exit(EXIT_FAILURE);
    }

    printf("free: %.f, total %.f\n", (double)avail/Mb, (double)total/Mb);

    int *buf_d = 0;
    size_t nwords = total / sizeof(int);
    size_t words_per_Mb = Mb / sizeof(int);

    while (cudaMalloc((void**)&buf_d,  nwords * sizeof(int)) == cudaErrorMemoryAllocation) {
      cudaFree(buf_d);
      nwords -= words_per_Mb;
      if (nwords < words_per_Mb) {
        // signal no free memory
        break;
      }
    }
    cudaFree(buf_d);

    printf("can allocate:  %.fMB\n", (double)nwords/words_per_Mb);

    //sleep(1000);  /* keep consuming RAM */

    return 0;

}
