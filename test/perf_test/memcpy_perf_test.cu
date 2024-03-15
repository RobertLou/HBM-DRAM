/*
Target: Copy Device data to Host
        Compare direct copy or using memcpy
*/
#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "../../time/timecalculate.h"

__global__ void copy(int *dst, int *src, int N){
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i < N){
        dst[i] = src[i];
    }
}


int main(){
    const int N = 100000000;
    int *h_src, *d_src;
    int *h_dst1, *d_dst1;
    int *h_dst2;

    cudaMalloc((void **)&d_src, N * sizeof(int));
    cudaMalloc((void **)&d_dst1, N * sizeof(int));
    
    cudaHostAlloc((void **)&h_src, N * sizeof(int), cudaHostAllocDefault);
    cudaHostAlloc((void **)&h_dst1, N * sizeof(int), cudaHostAllocDefault);
    cudaHostAlloc((void **)&h_dst2, N * sizeof(int), cudaHostAllocDefault);
    for (int i = 0; i < N;++i){
        h_src[i] = i;
    }
    cudaMemcpy(d_src, h_src, N * sizeof(int), cudaMemcpyHostToDevice);
    CTimeCalculate iTimeCal;

    iTimeCal.StartWork("Copy to Device then copy to Host");
    copy<<<(N + 127) / 128, 128>>>(d_dst1, d_src, N);
    cudaMemcpyAsync(h_dst1, d_dst1, N * sizeof(int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    iTimeCal.EndWork("Copy to Device then copy to Host");
    
    iTimeCal.StartWork("Direct Copy to Host");
    copy<<<(N + 127) / 128, 128>>>(h_dst2, d_src, N);
    cudaDeviceSynchronize();
    iTimeCal.EndWork("Direct Copy to Host");

    std::cout << h_dst1[13] << std::endl;
    std::cout << h_dst2[13] << std::endl;

    cudaFree(d_src);
    cudaFree(d_dst1);
    
    cudaFreeHost(h_src);
    cudaFreeHost(h_dst1);
    cudaFreeHost(h_dst2);
    
    return 0;
}