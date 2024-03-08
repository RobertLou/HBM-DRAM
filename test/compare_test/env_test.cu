#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void add(int *global_counter){
    atomicAdd(global_counter, 1);
}

int main(){
    int h_global_counter, *d_global_counter;
    int size = 12968;

    cudaMalloc((void**)&d_global_counter, sizeof(int));
    
    for(int i = 0;i < size;i++){
        add<<<1, 1>>>(d_global_counter);
    }

    cudaDeviceSynchronize();
    cudaMemcpy(&h_global_counter, d_global_counter, sizeof(int), cudaMemcpyDeviceToHost);
    std::cout << h_global_counter << std::endl;

    int runtime_version;
    cudaRuntimeGetVersion(&runtime_version);
    std::cout << "CUDA Runtime Version: " << runtime_version << std::endl;
    return 0;
}