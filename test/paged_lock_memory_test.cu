#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "../time/timecalculate.h"

#define SIZE (100 * 1024 * 1024)  // 100M
#define N 1

__global__ void Init(int *a, int length){
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i < length){
        a[i] = i;
    }
}


float cuda_malloc_test(int size, bool up){
    cudaEvent_t start, stop;
    int *a, *dev_a;
    float elapsedTime;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    a = (int *)malloc(size * sizeof(int));
    cudaMalloc((void **)&dev_a, size * sizeof(int));

    cudaEventRecord(start, 0);
    for(int i = 0; i < N;i++){
        if(up){
            cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
        }
        else{
            cudaMemcpy(a, dev_a, size * sizeof(int), cudaMemcpyDeviceToHost);
        }
    }

    int block_dim = 128;
    Init<<<(size + block_dim - 1) / block_dim, block_dim>>>(dev_a, size);
   
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    free(a);
    cudaFree(dev_a);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return elapsedTime;
}

float host_malloc_test(int size, bool up){
    cudaEvent_t start, stop;
    int *a, *dev_a;
    float elapsedTime;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);


    cudaHostAlloc((void **)&a, size * sizeof(int), cudaHostAllocDefault);
    cudaMalloc((void **)&dev_a, size * sizeof(int));
    
    cudaEventRecord(start, 0);
    for(int i = 0; i < N;i++){
        if(up){
            cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
        }
        else{
            cudaMemcpy(a, dev_a, size * sizeof(int), cudaMemcpyDeviceToHost);
        }
    }

    int block_dim = 128;
    Init<<<(size + block_dim - 1) / block_dim, block_dim>>>(dev_a, size);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    cudaFreeHost(a);
    cudaFree(dev_a);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return elapsedTime;
}

float host_malloc_test2(int size, bool up){
    cudaEvent_t start, stop;
    int *a;
    float elapsedTime;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaHostAlloc((void **)&a, size * sizeof(int), cudaHostAllocDefault);
    
    cudaEventRecord(start, 0);

    int block_dim = 128;
    Init<<<(size + block_dim - 1) / block_dim, block_dim>>>(a, size);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    cudaFreeHost(a);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return elapsedTime;
}

float cuda_malloc_host_test(int size, bool up){
    cudaEvent_t start, stop;
    int *a, *dev_a;
    float elapsedTime;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);


    cudaMallocHost((void **)&a, size * sizeof(int));
    cudaMalloc((void **)&dev_a, size * sizeof(int));
    
    cudaEventRecord(start, 0);
    for(int i = 0; i < N;i++){
        if(up){
            cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
        }
        else{
            cudaMemcpy(a, dev_a, size * sizeof(int), cudaMemcpyDeviceToHost);
        }
    }

    int block_dim = 128;
    Init<<<(size + block_dim - 1) / block_dim, block_dim>>>(dev_a, size);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    cudaFreeHost(a);
    cudaFree(dev_a);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return elapsedTime;
}

int main(){
    float elapsedTime;
    float MB = (float) 100 * SIZE * sizeof(int) / 1024 / 1024;
    CTimeCalculate iTimeCal;

    iTimeCal.StartWork("Cuda malloc");
    elapsedTime = cuda_malloc_test(SIZE, true);
    printf("elapsedTime: %3.1f ms\n", elapsedTime);
    elapsedTime = cuda_malloc_test(SIZE, false);
    printf("elapsedTime: %3.1f ms\n", elapsedTime);
    iTimeCal.EndWork("Cuda malloc");

    iTimeCal.StartWork("host malloc1");
    elapsedTime = host_malloc_test(SIZE, true);
    printf("elapsedTime: %3.1f ms\n", elapsedTime);
    elapsedTime = host_malloc_test(SIZE, false);
    printf("elapsedTime: %3.1f ms\n", elapsedTime);
    iTimeCal.EndWork("host malloc1");

    iTimeCal.StartWork("host malloc2");
    elapsedTime = host_malloc_test2(SIZE, true);
    printf("elapsedTime: %3.1f ms\n", elapsedTime);
    elapsedTime = host_malloc_test2(SIZE, false);
    printf("elapsedTime: %3.1f ms\n", elapsedTime);
    iTimeCal.EndWork("host malloc2");

    iTimeCal.StartWork("host malloc3");
    elapsedTime = cuda_malloc_host_test(SIZE, true);
    printf("elapsedTime: %3.1f ms\n", elapsedTime);
    elapsedTime = cuda_malloc_host_test(SIZE, false);
    printf("elapsedTime: %3.1f ms\n", elapsedTime);
    iTimeCal.EndWork("host malloc3");

    return 0;
}