#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "../time/timecalculate.h"

__global__ void add(int *a, int *b, int* c, int N){
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i < N){
        c[i] = a[i] + b[i];
    }
}


int main(){
    int *a, *b, *c;
    int *dev_c;
    CTimeCalculate iTimeCal;

    int size = 1000000;

    cudaHostAlloc((void **)&a, size * sizeof(int), cudaHostAllocDefault);
    cudaHostAlloc((void **)&b, size * sizeof(int), cudaHostAllocDefault);
    cudaHostAlloc((void **)&c, size * sizeof(int), cudaHostAllocDefault);
    cudaMalloc((void **)&dev_c, size * sizeof(int));

    for(int i = 0; i < size; i++){
        a[i] = 1;
        b[i] = 2;
    }

    

    cudaStream_t myStream;
    cudaStreamCreate(&myStream);


    iTimeCal.StartWork("Kernel Function");
    add<<<(size + 127) / 128, 128, 0, myStream>>>(a, b, dev_c, size);
    cudaMemcpyAsync(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost, myStream);
    cudaStreamSynchronize(myStream);
    for(int i = 0; i < size; i++){
        if(c[i] != 3){
            std::cout << c[i] << std::endl;
            std::cout << i << std::endl;
        }
    }
    iTimeCal.EndWork("Kernel Function");

    


    cudaFreeHost(a);
    cudaFreeHost(b);
    cudaFreeHost(c);
    cudaFree(dev_c);
    cudaStreamDestroy(myStream);
    return 0;
}