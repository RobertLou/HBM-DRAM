#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "../time/timecalculate.h"

#define MAX_THREADS_PER_BLOCK 128
#define MAX_ELEMENTS_PER_BLOCK (MAX_THREADS_PER_BLOCK * 2)

__global__ void parallel_large_scan_kernel(int *data, int *prefix_sum, int N, int *sums)
{
    __shared__ int tmp[MAX_ELEMENTS_PER_BLOCK];
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int block_offset = bid * MAX_ELEMENTS_PER_BLOCK;
    int leaf_num = MAX_ELEMENTS_PER_BLOCK;

    tmp[tid * 2] = tid * 2 + block_offset < N ? data[tid * 2 + block_offset] : 0;
    tmp[tid * 2 + 1] = tid * 2 + 1 + block_offset < N ? data[tid * 2 + 1 + block_offset] : 0;
    __syncthreads();

    int offset = 1;
    for (int d = leaf_num >> 1; d > 0; d >>= 1)
    {
        if (tid < d)
        {
            int ai = offset * (2 * tid + 1) - 1;
            int bi = offset * (2 * tid + 2) - 1;
            tmp[bi] += tmp[ai];
        }
        offset *= 2;
        __syncthreads();
    }

    if (tid == 0)
    {
        sums[bid] = tmp[leaf_num - 1];
        tmp[leaf_num - 1] = 0;
    }
    __syncthreads();

    for (int d = 1; d < leaf_num; d *= 2)
    {
        offset >>= 1;
        if (tid < d)
        {
            int ai = offset * (2 * tid + 1) - 1;
            int bi = offset * (2 * tid + 2) - 1;

            float v = tmp[ai];
            tmp[ai] = tmp[bi];
            tmp[bi] += v;
        }
        __syncthreads();
    }

    if (tid * 2 + block_offset < N)
    {
        prefix_sum[tid * 2 + block_offset] = tmp[tid * 2];
    }
    if (tid * 2 + 1 + block_offset < N)
    {
        prefix_sum[tid * 2 + 1 + block_offset] = tmp[tid * 2 + 1];
    }
}

__global__ void add_kernel(int *prefix_sum, int *value, int N)
{
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int block_offset = bid * MAX_ELEMENTS_PER_BLOCK;
    int ai = tid + block_offset;
    int bi = tid + (MAX_ELEMENTS_PER_BLOCK >> 1) + block_offset;

    if (ai < N)
    {
        prefix_sum[ai] += value[bid];
    }
    if (bi < N)
    {
        prefix_sum[bi] += value[bid];
    }
}

void recursive_scan(int *d_data, int *d_prefix_sum, int N)
{
    int block_num = (N - 1) / MAX_ELEMENTS_PER_BLOCK + 1;
    int *d_sums, *d_sums_prefix_sum;  // 用来保存block数组和、数组和的前缀和
    cudaMalloc(&d_sums, block_num * sizeof(int));
    cudaMalloc(&d_sums_prefix_sum, block_num * sizeof(int));

    parallel_large_scan_kernel<<<block_num, MAX_THREADS_PER_BLOCK>>>(d_data, d_prefix_sum, N, d_sums);

    if (block_num != 1)
    {
        recursive_scan(d_sums, d_sums_prefix_sum, block_num);
        add_kernel<<<block_num, MAX_THREADS_PER_BLOCK>>>(d_prefix_sum, d_sums_prefix_sum, N);
    }
}


int main(){
    int *in, *out;
    int *d_in, *d_out;
    CTimeCalculate iTimeCal;

    int size = 111111;

    in = (int *)malloc(sizeof(int) * size);
    out = (int *)malloc(sizeof(int) * (size + 1));
    
    cudaMalloc((void**)&d_in, sizeof(int) * size);
    cudaMalloc((void**)&d_out, sizeof(int) * (size + 1));

    for(int i = 0; i < size; i++){
        in[i] = 1;
    }

    cudaMemcpy(d_in, in, sizeof(int) * size, cudaMemcpyHostToDevice);


    iTimeCal.StartWork("Kernel Function");
    recursive_scan(d_in, d_out, size + 1);
    cudaDeviceSynchronize();
    iTimeCal.EndWork("Kernel Function");


    cudaMemcpy(out, d_out, sizeof(int) * (size + 1), cudaMemcpyDeviceToHost);

    for(int i = 0; i <= size; i++){
        if(out[i] != i){
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