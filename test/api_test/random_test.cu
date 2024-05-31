#include <iostream>
#include <curand_kernel.h>

#define N 256 // 定义线程数量
#define THRESHOLD 0.5 // 定义随机数阈值

__global__ void generateAndExecute(float threshold) {
    // 获取全局线程索引
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    // 初始化CURAND状态
    curandState state;
    curand_init(1234, idx, 0, &state); // 种子、线程ID、偏移量、状态

    // 生成0到1之间的随机浮点数
    float randVal = curand_uniform(&state);

    // 根据随机数的值执行代码
    if (randVal > threshold) {
        printf("Thread %d: Random value %.4f > %.4f, executing code\n", idx, randVal, threshold);
        // 在这里执行你的代码
    } else {
        printf("Thread %d: Random value %.4f <= %.4f, not executing code\n", idx, randVal, threshold);
    }
}

int main() {
    // 启动CUDA内核
    generateAndExecute<<<1, N>>>(THRESHOLD);

    // 等待CUDA内核执行完成
    cudaDeviceSynchronize();

    return 0;
}
