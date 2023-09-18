#include <iostream>
#include <cooperative_groups.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define WARP_SIZE 32

namespace cg = cooperative_groups;

__global__ void add(int *a, int *b, int N){
    cg::thread_block_tile<WARP_SIZE> warp_tile =
        cg::tiled_partition<WARP_SIZE>(cg::this_thread_block());
    const size_t lane_idx = warp_tile.thread_rank();
    
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i < N){
        a[i] = i;
        b[i] = warp_tile.meta_group_rank();
    }
}


int main(){

    int *a, *b;
    int *dev_a, *dev_b;


    int size = 1000;

    a = (int *)malloc(sizeof(int) * size);
    b = (int *)malloc(sizeof(int) * size);
    
    cudaMalloc((void**)&dev_a, sizeof(int) * size);
    cudaMalloc((void**)&dev_b, sizeof(int) * size);

    for(int i = 0; i < size; i++){
        a[i] = 1;
        b[i] = 2;
    }

    cudaMemcpy(dev_a, a, sizeof(int) * size, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, sizeof(int) * size, cudaMemcpyHostToDevice);


    add<<<(size + 127) / 128, 128>>>(dev_a, dev_b, size);
    cudaDeviceSynchronize();



    cudaMemcpy(a, dev_a, sizeof(int) * size, cudaMemcpyDeviceToHost);
    cudaMemcpy(b, dev_b, sizeof(int) * size, cudaMemcpyDeviceToHost);

    for(int i = 0; i < size; i++){
        std::cout << a[i] << "," << b[i] << std::endl;
    }
    return 0;
}