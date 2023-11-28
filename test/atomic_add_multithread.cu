#include <iostream>
#include <thread>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void add(int *global_counter, int N){
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i < N){
        atomicAdd(global_counter, 1);
    }
}

void Add(int *global_counter, int N) {
    add<<<(N + 127) / 128, 128>>>(global_counter, N);
}


int main(){

    int *h_global_counter, *d_global_counter;
    int size = 10000;

    h_global_counter = (int *)malloc(sizeof(int));
    cudaMalloc((void**)&d_global_counter, sizeof(int));

    *h_global_counter = 0;
    cudaMemcpy(d_global_counter, h_global_counter, sizeof(int), cudaMemcpyHostToDevice);

    std::thread thread1(Add, d_global_counter, size);
    std::thread thread2(Add, d_global_counter, size);
    std::thread thread3(Add, d_global_counter, size);
    std::thread thread4(Add, d_global_counter, size);

    thread1.join();
    thread2.join();
    thread3.join();
    thread4.join();

    cudaDeviceSynchronize();
    cudaMemcpy(h_global_counter, d_global_counter, sizeof(int), cudaMemcpyDeviceToHost);

    std::cout << *h_global_counter << std::endl;
    return 0;
}