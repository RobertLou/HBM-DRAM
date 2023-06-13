#include "embedding_map.h"

__global__ void UpdateOneEmbedding(Parameters **deviceAddressBatch, int currentBatchSize){
	int i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i < currentBatchSize){
        for(int j = 0;j < EMBEDDING_DIM;j++){
            deviceAddressBatch[i]->a[j] += g * g;
            deviceAddressBatch[i]->v[j] -= (c * g * 1.0) / sqrt(deviceAddressBatch[i]->a[j]);
        }
    }

}

__global__ void GatherEmbedding(Parameters **deviceAddressBatch, Parameters *devicegatherResult, int currentBatchSize){
	int i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i < currentBatchSize){
        for(int j = 0;j < EMBEDDING_DIM;j++){
            devicegatherResult[i].a[j] = deviceAddressBatch[i]->a[j];
            devicegatherResult[i].v[j] = deviceAddressBatch[i]->v[j];
        }
    }
}

Parameters* CEmbeddingMap::Get(int Key) {
    std::lock_guard<std::mutex> guard(a_mutex);
    return a_map.at(Key);
};

void CEmbeddingMap::Set(int Key, Parameters* Value) {
    std::lock_guard<std::mutex> guard(a_mutex);
    a_map.insert(std::make_pair(Key, Value)); 
};

void CEmbeddingMap::Erase(int Key)
{
    std::lock_guard<std::mutex> guard(a_mutex);
    a_map.erase(Key);
}

void CEmbeddingMap::InitEmbedding(std::string strFileloc,std::vector<Parameters> &line,int bFirstLineDelete){
    std::ifstream ifDataSet;
    ifDataSet.open(strFileloc);

    std::string strLine;
    char cComma;
    int nKeyTmp;
    std::vector<int> vKey;
    
    if(bFirstLineDelete){
        std::getline(ifDataSet, strLine);
    }
    float a_f,v_f;
    while (std::getline(ifDataSet, strLine))
    {
        std::stringstream ss(strLine);
        Parameters tmp;
        ss >> nKeyTmp;
        ss >> cComma;
        ss >> a_f;
        ss >> cComma;
        ss >> v_f;
        for(int i = 0;i < EMBEDDING_DIM;++i){
            tmp.a[i] = a_f;
            tmp.v[i] = v_f;
        }
        line.emplace_back(tmp);
        vKey.emplace_back(nKeyTmp);
    }

    int length = line.size();

    cudaMalloc((void **)&GPUEmbeddingAddress, length * sizeof(Parameters));
    cudaMemcpy(GPUEmbeddingAddress, &line[0], length * sizeof(Parameters), cudaMemcpyHostToDevice);
    
    int i = 0;
    for (auto iter1 = vKey.begin(); iter1 != vKey.end(); iter1++) {
        Set(*iter1, GPUEmbeddingAddress + i);
        i++;
    }

    ifDataSet.close();
}

void CEmbeddingMap::UpdateBatch(const std::vector<int>& line, int nCursor, Parameters **addressBatch, int currentBatchSize, TimeInterval &ti){
    int nBatchCursor = 0;

    //查询key所对应的GPU地址
    clock_gettime(CLOCK_MONOTONIC, &ti.tMemStart);
    for (auto iter = line.cbegin() + nCursor; iter != line.cbegin() + nCursor + currentBatchSize; iter++) {
        addressBatch[nBatchCursor] = Get(*iter);
        nBatchCursor++;
    }
    clock_gettime(CLOCK_MONOTONIC, &ti.tMemEnd);
    ti.fMemcpyTime1 += ((double)(ti.tMemEnd.tv_sec - ti.tMemStart.tv_sec)*1000000000 + ti.tMemEnd.tv_nsec - ti.tMemStart.tv_nsec)/1000000;


    //将embedding所对应的GPU地址拷贝到GPU
    Parameters **deviceAddressBatch;
    cudaMalloc((void **)&deviceAddressBatch, currentBatchSize * sizeof(Parameters *));
    cudaMemcpy(deviceAddressBatch, addressBatch, currentBatchSize * sizeof(Parameters *), cudaMemcpyHostToDevice);

    //计算更新embedding
    UpdateOneEmbedding<<<BATCH_SIZE/nDimBlock, nDimBlock>>>(deviceAddressBatch, currentBatchSize);
}

void CEmbeddingMap::UpdateWork(const std::vector<int>& line, int start, int end, int workerId)
	{	
		TimeInterval ti;
        int cursor = start;
        Parameters **addressBatch= new Parameters*[BATCH_SIZE];


		while(end - cursor >= BATCH_SIZE){
			UpdateBatch(line, cursor, addressBatch, BATCH_SIZE, ti);
			cursor += BATCH_SIZE;
		}
		UpdateBatch(line, cursor, addressBatch, end - cursor, ti);
		delete []addressBatch;

		std::cout << "线程" << workerId << "已经结束" << std::endl;
		std::cout << "memcpy time 1:" << ti.fMemcpyTime1 << "ms" << std::endl;		//CPU memcpy time
		//std::cout << "memcpy time 2:" << ti.fMemcpyTime2 << "ms" << std::endl;
	}

void CEmbeddingMap::MultiThreadUpdateEV(const std::vector<int>& line) {
    int scope = line.size() / THREAD_NUM;

    std::thread th_arr[THREAD_NUM];

    for (unsigned int i = 0; i < THREAD_NUM - 1; ++i) {
        th_arr[i] = std::thread(&CEmbeddingMap::UpdateWork, this, std::ref(line), i * scope, (i + 1) * scope, i);
    }
    th_arr[THREAD_NUM - 1] = std::thread(&CEmbeddingMap::UpdateWork, this, std::ref(line), (THREAD_NUM - 1) * scope, line.size(), THREAD_NUM - 1);
    for (unsigned int i = 0; i < THREAD_NUM; ++i) {
        th_arr[i].join();
    }
}

void CEmbeddingMap::GatherBatch(const std::vector<int>& line, int cursor, Parameters *gatherResult, int currentBatchSize){ 
    int nBatchCursor = 0;
    Parameters **addressBatch= new Parameters*[BATCH_SIZE];

    //查询key所对应的GPU地址
    for (auto iter = line.cbegin() + cursor; iter != line.cbegin() + cursor + currentBatchSize; iter++) {
        addressBatch[nBatchCursor] = Get(*iter);
        nBatchCursor++;
    }

    //将embedding所对应的GPU地址拷贝到GPU
    Parameters **deviceAddressBatch;
    cudaMalloc((void **)&deviceAddressBatch, currentBatchSize * sizeof(Parameters *));
    cudaMemcpy(deviceAddressBatch, addressBatch, currentBatchSize * sizeof(Parameters *), cudaMemcpyHostToDevice);

    //创建查找到的embedding数据存储的空间
    Parameters *devicegatherResult;
    cudaMalloc((void **)&devicegatherResult, currentBatchSize * sizeof(Parameters));

    //Gather 
    GatherEmbedding<<<BATCH_SIZE/nDimBlock, nDimBlock>>>(deviceAddressBatch, devicegatherResult, currentBatchSize);
    cudaDeviceSynchronize();

    //将结果拷贝回CPU检验
    cudaMemcpy(&gatherResult[cursor], devicegatherResult, currentBatchSize * sizeof(Parameters), cudaMemcpyDeviceToHost);
    cudaFree(devicegatherResult);
    cudaFree(deviceAddressBatch);
    delete []addressBatch;
}

void CEmbeddingMap::GatherWork(const std::vector<int>& line, Parameters *gatherResult, int start, int end, int workerId){
    int cursor = start;

    while(end - cursor >= BATCH_SIZE){
        GatherBatch(line, cursor, gatherResult, BATCH_SIZE);
        cursor += BATCH_SIZE;
    }
    GatherBatch(line, cursor, gatherResult, end - cursor);
}

void CEmbeddingMap::MultiThreadGatherEV(const std::vector<int>& line, Parameters *gatherResult) {
    int scope = line.size() / THREAD_NUM;
    std::thread th_arr[THREAD_NUM];

    for (unsigned int i = 0; i < THREAD_NUM - 1; ++i) {
        th_arr[i] = std::thread(&CEmbeddingMap::GatherWork, this, std::ref(line), gatherResult, i * scope, (i + 1) * scope, i);
    }
    th_arr[THREAD_NUM - 1] = std::thread(&CEmbeddingMap::GatherWork, this, std::ref(line), gatherResult, (THREAD_NUM - 1) * scope, line.size(), THREAD_NUM - 1);
    for (unsigned int i = 0; i < THREAD_NUM; ++i) {
        th_arr[i].join();
    }
}

void CEmbeddingMap::DeleteEmbedding(){
    cudaFree(GPUEmbeddingAddress);
}