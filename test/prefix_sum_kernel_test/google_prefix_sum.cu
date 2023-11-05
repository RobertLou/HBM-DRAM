#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "../../time/timecalculate.h"

__global__ void prescan(int *g_odata, int *g_idata, int n)
{
    extern __shared__ int temp[2048]; // allocated on invocation
    int thid = threadIdx.x;
    int offset = 1;
    temp[2 * thid] = g_idata[2 * thid]; // load input into shared memory
    temp[2 * thid + 1] = g_idata[2 * thid + 1];
    for (int d = n >> 1; d > 0; d >>= 1) // build sum in place up the tree
    {
        __syncthreads();
        if (thid < d)
        {
            int ai = offset * (2 * thid + 1) - 1;
            int bi = offset * (2 * thid + 2) - 1;
            temp[bi] += temp[ai];
        }
        offset *= 2;
    }
    if (thid == 0)
    {
        temp[n - 1] = 0;
    }                              // clear the last element
    for (int d = 1; d < n; d *= 2) // traverse down tree & build scan
    {
        offset >>= 1;
        __syncthreads();
        if (thid < d)
        {
            int ai = offset * (2 * thid + 1) - 1;
            int bi = offset * (2 * thid + 2) - 1;
            int t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
    }
    __syncthreads();
    g_odata[2 * thid] = temp[2 * thid]; // write results to device memory
    g_odata[2 * thid + 1] = temp[2 * thid + 1];
}

int main(){
    int *in, *out;
    int *d_in, *d_out;
    CTimeCalculate iTimeCal;

    int size = 1024;

    in = (int *)malloc(sizeof(int) * size);
    out = (int *)malloc(sizeof(int) * size);
    
    cudaMalloc((void**)&d_in, sizeof(int) * size);
    cudaMalloc((void**)&d_out, sizeof(int) * size);

    for(int i = 0; i < size; i++){
        in[i] = 1;
    }

    cudaMemcpy(d_in, in, sizeof(int) * size, cudaMemcpyHostToDevice);


    iTimeCal.StartWork("Kernel Function");
    prescan<<<1, size>>>(d_out, d_in, size);
    cudaDeviceSynchronize();
    iTimeCal.EndWork("Kernel Function");


    cudaMemcpy(out, d_out, sizeof(int) * size, cudaMemcpyDeviceToHost);

    for(int i = 0; i < size; i++){
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