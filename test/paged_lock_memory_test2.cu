#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "../time/timecalculate.h"

#define SIZE (1000 * 1024 * 1024)
#define N 1

const int block_dim = 128;

__global__ void Init(int *a, int length){
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i < length){
        a[i] = i;
    }
}

void cuda_malloc_test(int size, bool up){
    CTimeCalculate iTimeCal;
    int *a, *dev_a;

    a = (int *)malloc(size * sizeof(int));
    cudaMalloc((void **)&dev_a, size * sizeof(int));
    for(int i = 0; i < size;i++){
        a[i] = 0;
    }

    iTimeCal.StartWork("Cuda malloc");
    cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    Init<<<(size + block_dim - 1) / block_dim, block_dim>>>(dev_a, size);
    cudaDeviceSynchronize();
    iTimeCal.EndWork("Cuda malloc");

    free(a);
    cudaFree(dev_a);
}

void host_malloc_test(int size, bool up){
    CTimeCalculate iTimeCal;
    int *a, *dev_a;

    cudaHostAlloc((void **)&a, size * sizeof(int), cudaHostAllocDefault);
    cudaMalloc((void **)&dev_a, size * sizeof(int));
    for(int i = 0; i < size;i++){
        a[i] = 0;
    }
    
    iTimeCal.StartWork("host malloc");
    cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    Init<<<(size + block_dim - 1) / block_dim, block_dim>>>(dev_a, size);
    cudaDeviceSynchronize();
    iTimeCal.EndWork("host malloc");

    cudaFreeHost(a);
    cudaFree(dev_a);
}

void host_malloc_test2(int size, bool up){
    CTimeCalculate iTimeCal;
    int *a;

    cudaHostAlloc((void **)&a, size * sizeof(int), cudaHostAllocDefault);
    for(int i = 0; i < size;i++){
        a[i] = 0;
    }

    iTimeCal.StartWork("host malloc");
    Init<<<(size + block_dim - 1) / block_dim, block_dim>>>(a, size);
    cudaDeviceSynchronize();
    std::cout << a[3] << std::endl;
    iTimeCal.EndWork("host malloc");

    cudaFreeHost(a);
}

int main(){
    int *b;
    cudaMalloc((void **)&b, sizeof(int));
    cudaFree(b);
    cudaHostAlloc((void **)&b, sizeof(int), cudaHostAllocDefault);
    cudaFreeHost(b);

    //cuda_malloc_test(SIZE, true);
    //host_malloc_test(SIZE, true);
    host_malloc_test2(SIZE, true);

    return 0;
}