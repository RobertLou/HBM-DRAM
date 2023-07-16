#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void write(int *count, int *a, int length){
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    __shared__ int lock;
    if(i == 0)
        lock = 0;
    if(i < length){
        bool blocked = true;
        while(blocked) {
            if(0 == atomicCAS(&lock, 0, 1)) {
                atomicAdd(count, 1);
                a[i] = *count;
                atomicExch(&lock, 0);
                blocked = false;
            }
        }
    }
}

int main(){
    int *a, *count, *b;
    int *dev_a, *dev_count;


    a = (int *)malloc(sizeof(int) * 100);
    b = (int *)malloc(sizeof(int) * 100);
    count = (int *)malloc(sizeof(int) * 1);

    
    cudaMalloc((void**)&dev_a, sizeof(int) * 100);
    cudaMalloc((void**)&dev_count, sizeof(int) * 1);

    *count = 0;
    for(int i = 0; i < 100; i++){
        a[i] = 0;
    }
    cudaMemcpy(dev_a, a, sizeof(int) * 100, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_count, count, sizeof(int) * 1, cudaMemcpyHostToDevice);

    write<<<(100 + 31) / 32, 32>>>(dev_count, dev_a, 100);

    cudaMemcpy(b, dev_a, sizeof(int) * 100, cudaMemcpyDeviceToHost);
    for(int i = 0; i < 100; i++){
        std::cout << b[i] << std::endl;
    }

    return 0;
}