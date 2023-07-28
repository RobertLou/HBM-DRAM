#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void write(int *count, int *a, int length, int *lock){
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    //__shared__ int lock;
    if(i == 0)
        *lock = 0;
    if(i < length){
        bool blocked = true;
        while(blocked) {
            if(0 == atomicCAS(lock, 0, 1)) {
                //*count = *count + 1;
                atomicAdd(count, 1);
                a[i] = *count;
                atomicExch(lock, 0);
                blocked = false;
            }
        }
    }
}

int main(){
    int *a, *count, *b;
    int *dev_a, *dev_count;

    int length = 10000;
    int *lock;

    a = (int *)malloc(sizeof(int) * length);
    b = (int *)malloc(sizeof(int) * length);
    count = (int *)malloc(sizeof(int) * 1);

    
    cudaMalloc((void**)&dev_a, sizeof(int) * length);
    cudaMalloc((void**)&dev_count, sizeof(int) * 1);
    cudaMalloc((void**)&lock, sizeof(int) * 1);

    *count = 0;
    for(int i = 0; i < length; i++){
        a[i] = 0;
    }
    cudaMemcpy(dev_a, a, sizeof(int) * length, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_count, count, sizeof(int) * 1, cudaMemcpyHostToDevice);

    write<<<(length + 31) / 32, 32>>>(dev_count, dev_a, length, lock);

    cudaMemcpy(b, dev_a, sizeof(int) * length, cudaMemcpyDeviceToHost);
    for(int i = 0; i < length; i++){
        std::cout << b[i] << std::endl;
    }

    return 0;
}