#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "../time/timecalculate.h"

int main(){
    int *a, *b;
    int *dev_a, *dev_b;
    CTimeCalculate iTimeCal;

    a = (int *)malloc(sizeof(int) * 1);
    b = (int *)malloc(sizeof(int) * 100000);
    
    cudaMalloc((void**)&dev_a, sizeof(int) * 1);
    cudaMalloc((void**)&dev_b, sizeof(int) * 100000);

    iTimeCal.StartWork("memcpy 1 int");
    cudaMemcpy(dev_a, a, sizeof(int) * 1, cudaMemcpyHostToDevice);
    iTimeCal.EndWork("memcpy 1 int");

    iTimeCal.StartWork("memcpy 100000 int");
    cudaMemcpy(dev_b, b, sizeof(int) * 100000, cudaMemcpyHostToDevice);
    iTimeCal.EndWork("memcpy 100000 int");
    return 0;
}