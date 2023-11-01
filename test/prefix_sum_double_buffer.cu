#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "../time/timecalculate.h"

__global__ void prefixSum(int* input, int* output, int n) {
    extern __shared__ int temp[2048];  // 共享内存用于存储中间结果
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int pout = 0, pin = 1;  // 用于交替输入/输出缓冲区
    temp[tid] = input[tid];  // 将输入数据复制到共享内存
    __syncthreads();

    for (int offset = 1; offset < n; offset *= 2) {
        pout = 1 - pout;  // 切换输入/输出缓冲区
        pin = 1 - pout;
        if (tid >= offset) {
            temp[pout * n + tid] = temp[pin * n + tid] + temp[pin * n + tid - offset];
        } else {
            temp[pout * n + tid] = temp[pin * n + tid];
        }
        __syncthreads();
    }
    output[tid] = temp[pout * n + tid];  // 将结果写回输出数组
}


int main(){
    int *in, *out;
    int *d_in, *d_out;
    CTimeCalculate iTimeCal;

    int size = 128;

    in = (int *)malloc(sizeof(int) * size);
    out = (int *)malloc(sizeof(int) * size);
    
    cudaMalloc((void**)&d_in, sizeof(int) * size);
    cudaMalloc((void**)&d_out, sizeof(int) * size);

    for(int i = 0; i < size; i++){
        in[i] = 1;
    }

    cudaMemcpy(d_in, in, sizeof(int) * size, cudaMemcpyHostToDevice);


    iTimeCal.StartWork("Kernel Function");
    prefixSum<<<1, size>>>(d_in, d_out, size);
    cudaDeviceSynchronize();
    iTimeCal.EndWork("Kernel Function");


    cudaMemcpy(out, d_out, sizeof(int) * size, cudaMemcpyDeviceToHost);

    for(int i = 0; i < size; i++){
        if(out[i] != i + 1){
            std::cout << i << std::endl;
            std::cout << out[i] << std::endl;
        }
    }

    cudaFree(d_in);
    cudaFree(d_out);
    delete[] in;
    delete[] out;
    return 0;
}