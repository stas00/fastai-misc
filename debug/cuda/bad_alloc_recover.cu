// how to recover from bad_alloc
// from https://stackoverflow.com/questions/32121827/cant-restore-my-gpu-after-bad-alloc-with-cudadevicereset-from-the-cuda-libr
 //
 // any error by a Cuda runtime call is registered internally and can be viewed using either cudaPeekAtLastError() or cudaGetLastError(). The first can be called multiple times to read the same error, while the later is used to READ AND CLEAR the error. It is advisable to clear the earlier error before making subsequent Cuda runtime call, using cudaGetLastError().


#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <cuda.h>
#include <thrust/system_error.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>

#define CUDA_CALL(x)do { if((x) != cudaSuccess) { return -11;}} while(0)

typedef typename thrust::device_vector<size_t>  tDevVecInt;
typedef typename thrust::device_vector<float>   tDevVecFlt;

struct modSim : public thrust::unary_function<int, int>
{
    int szMat;
    int p;

    modSim(int in1, int in2)
    {
        this->p = in1;
        this->szMat = in2;
    }
    __host__ __device__ int operator()(const int &x)
    {
        return (x/szMat)*p+(x%p);
    }
};

 int test(size_t szData)
{

    modSim moduloCol(3, 33);

    CUDA_CALL(cudaSetDevice(0));

    try
    {

        tDevVecFlt devRand(szData);
        tDevVecInt devIndices(szData);
        tDevVecFlt devData(szData);

        thrust::sequence(devRand.begin(), devRand.end());
        thrust::tabulate(devIndices.begin(), devIndices.end(), moduloCol);
        thrust::sort_by_key(devIndices.begin(), devIndices.end(), devRand.begin());

    }
    catch(std::bad_alloc &e)
    {
        std::cout << e.what() << std::endl;
        CUDA_CALL(cudaGetLastError()); // without this call all the following allocations will fail too https://stackoverflow.com/a/32265613/9201239
        CUDA_CALL(cudaDeviceReset());
        CUDA_CALL(cudaSetDevice(0));
        return -3;
    }
    catch(thrust::system_error &e)
    {
        std::cout << e.what() << std::endl;
        CUDA_CALL(cudaDeviceReset());
        CUDA_CALL(cudaSetDevice(0));
        return -2;
    }

    CUDA_CALL(cudaDeviceReset());
    return 0;
}


int main(void)
{

    size_t n;
    int retVal;

    n = 3000;
    retVal = test(n);
    std::cout << retVal << std::endl;

    n = 300000000;
    retVal = test(n);
    std::cout << retVal << std::endl;

    n = 3000;
    retVal = test(n);
    std::cout << retVal << std::endl;

    return(0);
}
