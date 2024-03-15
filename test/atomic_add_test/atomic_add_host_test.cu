#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void add1(int *global_counter, int N){
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i < N){
        atomicAdd(global_counter, 1);
    }
}

__global__ void add2(int *global_counter, int N){
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i < N){
        atomicAdd(global_counter, 1);
    }
}

int main(){
    int *h_global_counter, *d_global_counter;
    int *host_counter;
    int size = 10000;

    h_global_counter = (int *)malloc(sizeof(int));
    cudaMalloc((void**)&d_global_counter, sizeof(int));
    cudaHostAlloc((void **)&host_counter, sizeof(int), cudaHostAllocDefault);
    
    *h_global_counter = 0;
    *host_counter = 0;
    cudaMemcpy(d_global_counter, h_global_counter, sizeof(int), cudaMemcpyHostToDevice);

    for(int i = 0;i < 1000;i++){
        add1<<<(size + 127) / 128, 128>>>(d_global_counter, size);
    }

    for(int i = 0;i < 1000;i++){
        add2<<<(size + 127) / 128, 128>>>(host_counter, size);
    }

    cudaDeviceSynchronize();
    cudaMemcpy(h_global_counter, d_global_counter, sizeof(int), cudaMemcpyDeviceToHost);

    std::cout << *h_global_counter << std::endl;
    std::cout << *host_counter << std::endl;
    return 0;
}