#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void Init(void *a, int allocSize, int length, int EmbeddingDim){
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i < length){
        char *tmp_ptr = static_cast<char*>(a) + allocSize * i;
        int *key_ptr = static_cast<int *>(static_cast<void*>(tmp_ptr));
        int *freq_ptr = static_cast<int *>(static_cast<void*>(tmp_ptr + 4));
        int *value_ptr = static_cast<int *>(static_cast<void*>(tmp_ptr + 8));
        *key_ptr = i;
        *freq_ptr = 0;
        for (int j = 0; j < EmbeddingDim; j++){
            value_ptr[j] = j;
        }
    }

}

struct GPUHeader{
    int key;
    int frequency;
};

int main(){
    void *a;
    void *dev_a;

    const int EmbeddingDim = 32;
    const int CacheSize = 100;

    int allocSize = sizeof(GPUHeader) + EmbeddingDim * sizeof(int);

    a = malloc(allocSize * CacheSize);
    
    cudaMalloc((void**)&dev_a, allocSize * CacheSize);


    Init<<<(100 + 31) / 32, 32>>>(dev_a, allocSize, 100, EmbeddingDim);

    cudaMemcpy(a, dev_a, allocSize * CacheSize, cudaMemcpyDeviceToHost);
    for(int i = 0; i < 100; i++){
        char *tmp_ptr = static_cast<char*>(a) + allocSize * i;
        int *key_ptr = static_cast<int *>(static_cast<void*>(tmp_ptr));
        int *freq_ptr = static_cast<int *>(static_cast<void*>(tmp_ptr + 4));
        int *value_ptr = static_cast<int *>(static_cast<void*>(tmp_ptr + 8));
        std::cout << *key_ptr << std::endl;
        std::cout << *freq_ptr << std::endl;
        std::cout << value_ptr[0] << std::endl;
        std::cout << value_ptr[31] << std::endl;
    }

    return 0;
}