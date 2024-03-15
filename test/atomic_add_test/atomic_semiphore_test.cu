#include <iostream>
#include <cooperative_groups.h>
#include "cuda_runtime.h"
#include <cuda/std/atomic>
#include <cuda/std/semaphore>
#include "device_launch_parameters.h"

namespace cg = cooperative_groups;

using atomic_ref_counter_type = cuda::atomic<int, cuda::thread_scope_device>;

__global__ void add(atomic_ref_counter_type *global_counter, int N){
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i < N){
        global_counter->fetch_add(1, cuda::std::memory_order_relaxed);
    }
}

__global__ void add2(int *global_counter, int N){
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i < N){
        atomicAdd(global_counter, 1);
    }
}


int main(){
    atomic_ref_counter_type *global_counter;
    atomic_ref_counter_type check_global_counter;

    int *h_global_counter, *d_global_counter;
    int size = 10000;

    cudaMalloc((void**)&global_counter, sizeof(atomic_ref_counter_type));

    h_global_counter = (int *)malloc(sizeof(int));
    cudaMalloc((void**)&d_global_counter, sizeof(int));

    *h_global_counter = 0;
    cudaMemcpy(d_global_counter, h_global_counter, sizeof(atomic_ref_counter_type), cudaMemcpyHostToDevice);
    

    add<<<(size + 127) / 128, 128>>>(global_counter, size);
    add2<<<(size + 127) / 128, 128>>>(d_global_counter, size);
    cudaDeviceSynchronize();

    cudaMemcpy(&check_global_counter, global_counter, sizeof(atomic_ref_counter_type), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_global_counter, d_global_counter, sizeof(int), cudaMemcpyDeviceToHost);

    std::cout << check_global_counter << std::endl;
    std::cout << *h_global_counter << std::endl;
    return 0;
}