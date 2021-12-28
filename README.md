# HBM-DRAM
实现处理embedding的HBM-DRAM混合存储
##CPU_POC部分：
多线程更新embedding向量，每次查找一个batch的embedding，将其拷贝放入连续空间后进行更新
