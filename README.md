# HBM-DRAM
实现处理embedding的HBM-DRAM混合存储
## 项目结构
-embedding_map 实现embedding的数据存储结构和初始化、插入、删除、批量更新等操作 \
-FileRW 实现对csv文件读取，并将embedding table暂存到csv中方便比对   \
-time 实现计时
### POC_CPU：
多线程更新embedding向量，每次查找一个batch的embedding，将其拷贝放入连续空间后进行更新 \
embedding.csv 中包含了初始的（k，embedding）对，从ad_feature.csv中获得一个访问序列，然后进行读取和更新。
### POC_GPU
更新embedding由GPU完成，其中计时由于采取事件方式开销大，故采取nvprof方式观察计时

