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
    int *dev_a, *dev_b, *dev_c;
    CTimeCalculate iTimeCal;

    int size = 1000000;

    a = (int *)malloc(sizeof(int) * size);
    b = (int *)malloc(sizeof(int) * size);
    c = (int *)malloc(sizeof(int) * size);
    
    cudaMalloc((void**)&dev_a, sizeof(int) * size);
    cudaMalloc((void**)&dev_b, sizeof(int) * size);
    cudaMalloc((void**)&dev_c, sizeof(int) * size);

    for(int i = 0; i < size; i++){
        a[i] = 1;
        b[i] = 2;
    }

    cudaMemcpy(dev_a, a, sizeof(int) * size, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, sizeof(int) * size, cudaMemcpyHostToDevice);


    iTimeCal.StartWork("Kernel Function");
    add<<<(size + 127) / 128, 128>>>(dev_a, dev_b, dev_c, size);
    cudaDeviceSynchronize();
    iTimeCal.EndWork("Kernel Function");


    cudaMemcpy(c, dev_c, sizeof(int) * size, cudaMemcpyDeviceToHost);

    for(int i = 0; i < size; i++){
        if(c[i] != 3){
            std::cout << c[i] << std::endl;
        }
    }
    return 0;
}