# HBM-DRAM
实现处理embedding的HBM-DRAM混合存储
### POC_CPU部分：
多线程更新embedding向量，每次查找一个batch的embedding，将其拷贝放入连续空间后进行更新
### POC_GPU_NO_TIME
更新embedding由GPU完成，其中计时由于采取事件方式开销大，故采取nvprof方式观察计时
