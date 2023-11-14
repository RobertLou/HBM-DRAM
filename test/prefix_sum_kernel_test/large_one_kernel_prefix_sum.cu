#include "../../time/timecalculate.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cooperative_groups.h>
#include <iostream>

#define MAX_THREADS_PER_BLOCK 256
#define MAX_ELEMENTS_PER_BLOCK (MAX_THREADS_PER_BLOCK * 2)

namespace cg = cooperative_groups;

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

            int v = tmp[ai];
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

    __threadfence();
    int block_num = (N - 1) / MAX_ELEMENTS_PER_BLOCK + 1;
    if (bid == 0 && block_num != 1)
    {
        __shared__ int tmp2[MAX_ELEMENTS_PER_BLOCK];

        tmp2[tid * 2] = tid * 2 < block_num ? sums[tid * 2] : 0;
        tmp2[tid * 2 + 1] = tid * 2 + 1 < block_num ? sums[tid * 2 + 1] : 0;
        __syncthreads();

        offset = 1;
        for (int d = leaf_num >> 1; d > 0; d >>= 1)
        {
            if (tid < d)
            {
                int ai = offset * (2 * tid + 1) - 1;
                int bi = offset * (2 * tid + 2) - 1;
                tmp2[bi] += tmp2[ai];
            }
            offset *= 2;
            __syncthreads();
        }

        if (tid == 0)
        {
            tmp2[leaf_num - 1] = 0;
        }
        __syncthreads();

        for (int d = 1; d < leaf_num; d *= 2)
        {
            offset >>= 1;
            if (tid < d)
            {
                int ai = offset * (2 * tid + 1) - 1;
                int bi = offset * (2 * tid + 2) - 1;

                int v = tmp2[ai];
                tmp2[ai] = tmp2[bi];
                tmp2[bi] += v;
            }
            __syncthreads();
        }

        if (tid * 2 < block_num)
        {
            sums[tid * 2] = tmp2[tid * 2];
        }
        if (tid * 2 + 1 < block_num)
        {
            sums[tid * 2 + 1] = tmp2[tid * 2 + 1];
        }
    }

    if (block_num > 1)
    {

        cg::grid_group grid = cg::this_grid();
        grid.sync();
        int ai = tid + block_offset;
        int bi = tid + (MAX_ELEMENTS_PER_BLOCK >> 1) + block_offset;

        if (ai < N)
        {
            prefix_sum[ai] += sums[bid];
        }
        if (bi < N)
        {
            prefix_sum[bi] += sums[bid];
        }
    }
}

void recursive_scan(int *d_data, int *d_prefix_sum, int N)
{
    int block_num = (N - 1) / MAX_ELEMENTS_PER_BLOCK + 1;
    int *d_sums; // 用来保存block数组和、数组和的前缀和

    cudaHostAlloc(&d_sums, block_num * sizeof(int), cudaHostAllocDefault);

    void *kernelArgs[] = {(void *)&d_data,
                          (void *)&d_prefix_sum,
                          (void *)&N,
                          (void *)&d_sums};
    cudaLaunchCooperativeKernel((void *)parallel_large_scan_kernel, block_num, MAX_THREADS_PER_BLOCK, kernelArgs);
    cudaFreeHost(d_sums);
}

int main()
{
    int *in, *out;
    CTimeCalculate iTimeCal;

    int size = 1 << 16 - 1;

    cudaHostAlloc(&in, sizeof(int) * size, cudaHostAllocDefault);
    cudaHostAlloc(&out, sizeof(int) * (size + 1), cudaHostAllocDefault);

    for (int i = 0; i < size; i++)
    {
        in[i] = 1;
    }

    iTimeCal.StartWork("Kernel Function");
    recursive_scan(in, out, size + 1);
    cudaDeviceSynchronize();
    iTimeCal.EndWork("Kernel Function");

    for (int i = 0; i <= size; i++)
    {
        if (out[i] != i)
        {
            std::cout << i << std::endl;
            std::cout << out[i] << std::endl;
        }
    }

    cudaFreeHost(in);
    cudaFreeHost(out);
    return 0;
}