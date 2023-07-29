#include "embedding_map.h"

Parameters* CEmbeddingMap::Get(int Key){
    std::shared_lock<std::shared_mutex> lock(a_mutex);
	return a_map.at(Key);
};

void CEmbeddingMap::Set(int Key, Parameters* Value){
    std::unique_lock<std::shared_mutex> lock(a_mutex);
	a_map.insert(std::make_pair(Key, Value)); 
};

void CEmbeddingMap::Erase(int key){
	std::unique_lock<std::shared_mutex> lock(a_mutex);
	a_map.erase(key);
}

__global__ void InitEmptyCache(Parameters *GPUEmbeddingAddress){
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    GPUEmbeddingAddress[i].key = -1;
}

__global__ void DeviceInitEmbedding(int *locks, Parameters *GPUEmbeddingAddress, Parameters *AllGPUEmbeddings, int length){
	int i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i < length){
        int key = AllGPUEmbeddings[i].key;
        int cache_id = key % CACHE_NUM;
        int possible_place = cache_id * WAYS;
        bool blocked = true;
        while(blocked) {
            if(0 == atomicCAS(&locks[cache_id], 0, 1)) {
                for(int j = 0;j < WAYS;j++){
                    if(GPUEmbeddingAddress[possible_place + j].key == -1){
                        GPUEmbeddingAddress[possible_place + j].key = key;
                        for(int k = 0; k < EMBEDDING_DIM; k++){
                            GPUEmbeddingAddress[possible_place + j].a[k] = AllGPUEmbeddings[i].a[k];
                            GPUEmbeddingAddress[possible_place + j].v[k] = AllGPUEmbeddings[i].v[k];
                        }
                        GPUEmbeddingAddress[possible_place + j].frequency = 0;
                        break;
                    }
                }
                atomicExch(&locks[cache_id], 0);
                blocked = false;
            }
        }
    }
}

/* __global__ void GatherEmbedding(int *keyBatch, Parameters *GPUEmbeddingAddress, Parameters *deviceGatherResult, int *missCount, int *missIndexList, int *missKeyList, int *lock, int limit){
	int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j;
    if (i == 0){
        *missCount = 0;
        *lock = 0;
    }
    if(i < limit){
        int key = keyBatch[i];
        int cache_id = key % CACHE_NUM;
        int possible_place = cache_id * WAYS;
        for(j = 0; j < WAYS; j++){
            if(GPUEmbeddingAddress[possible_place + j].key == key){
                deviceGatherResult[i].key = key;
                for(int k = 0; k < EMBEDDING_DIM; k++){
                    deviceGatherResult[i].a[k] = GPUEmbeddingAddress[possible_place + j].a[k];
                    deviceGatherResult[i].v[k] = GPUEmbeddingAddress[possible_place + j].v[k];
                }
                atomicAdd(&GPUEmbeddingAddress[possible_place + j].frequency, 1);
                break;
            }
            if(GPUEmbeddingAddress[possible_place + j].key == -1){
                bool blocked = true;
                while(blocked) {
                    if(0 == atomicCAS(lock, 0, 1)) {
                        __threadfence();
                        missKeyList[*missCount] = key;
                        missIndexList[*missCount] = i;
                        atomicAdd(missCount, 1);
                        __threadfence();
                        atomicExch(lock, 0);
                        blocked = false;
                    }
                }
                break;
            }
        }
        if(j == WAYS){
            bool blocked = true;
            while(blocked) {
                if(0 == atomicCAS(lock, 0, 1)) {
                    __threadfence();
                    missKeyList[*missCount] = key;
                    missIndexList[*missCount] = i;
                    atomicAdd(missCount, 1);
                    __threadfence();
                    atomicExch(lock, 0);
                    blocked = false;
                }
            }
        }
    }
} */

__global__ void GatherEmbedding(int *keyBatch, Parameters *GPUEmbeddingAddress, Parameters *deviceGatherResult, int *missCount, int *missIndexList, int *missKeyList, int *lock, int limit){
	int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j;
    if (i == 0){
        *missCount = 0;
        *lock = 0;
    }
    if(i < limit * EMBEDDING_DIM){
        int key_index = i / EMBEDDING_DIM;
        int embedding_index = i % EMBEDDING_DIM;
        int key = keyBatch[key_index];
        int cache_id = key % CACHE_NUM;
        int possible_place = cache_id * WAYS;

        for(j = 0; j < WAYS; j++){
            if(GPUEmbeddingAddress[possible_place + j].key == key){
                if(embedding_index == 0){
                    deviceGatherResult[key_index].key = key;
                    atomicAdd(&GPUEmbeddingAddress[possible_place + j].frequency, 1);
                }
                deviceGatherResult[key_index].a[embedding_index] = GPUEmbeddingAddress[possible_place + j].a[embedding_index];
                deviceGatherResult[key_index].v[embedding_index] = GPUEmbeddingAddress[possible_place + j].v[embedding_index];
                break;
            }
              
            if(embedding_index == 0 && GPUEmbeddingAddress[possible_place + j].key == -1){
                bool blocked = true;
                while(blocked) {
                    if(0 == atomicCAS(lock, 0, 1)) {
                        __threadfence();
                        missKeyList[*missCount] = key;
                        missIndexList[*missCount] = key_index;
                        atomicAdd(missCount, 1);
                        __threadfence();
                        atomicExch(lock, 0);
                        blocked = false;
                    }
                }
                break;
            }
        }
        if(embedding_index == 0  && j == WAYS){
            bool blocked = true;
            while(blocked) {
                if(0 == atomicCAS(lock, 0, 1)) {
                    __threadfence();
                    missKeyList[*missCount] = key;
                    missIndexList[*missCount] = key_index;
                    atomicAdd(missCount, 1);
                    __threadfence();
                    atomicExch(lock, 0);
                    blocked = false;
                }
            }
        }
    }
}

__global__ void GatherMissingEmbedding(int *locks, int *keyBatch, Parameters *GPUEmbeddingAddress, Parameters *deviceGatherResult, int *missIndexList, int *missKeyList, Parameters *deviceMissingEmbedding, int limit){
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i < limit){
        int key = missKeyList[i];
        int index = missIndexList[i];
        int cache_id = key % CACHE_NUM;
        int possible_place = cache_id * WAYS;

        //写入Result
        deviceGatherResult[index].key =  key;
        deviceGatherResult[index].frequency = 0;
        for(int k = 0; k < EMBEDDING_DIM; k++){
            deviceGatherResult[index].a[k] = deviceMissingEmbedding[i].a[k];
            deviceGatherResult[index].v[k] = deviceMissingEmbedding[i].v[k];
        }
        
        //更新Cache
        bool blocked = true;
        int minFreq = 99999;
        int minPlace = -1;
        while(blocked) {
            if(0 == atomicCAS(&locks[cache_id], 0, 1)) {
                //寻找可替换位置
                for(int j = 0;j < WAYS;j++){
                    if(GPUEmbeddingAddress[possible_place + j].frequency < minFreq){
                        minFreq = GPUEmbeddingAddress[possible_place + j].frequency;
                        minPlace = j;
                    }
                }

                //替换
                GPUEmbeddingAddress[possible_place + minPlace].key = key;
                GPUEmbeddingAddress[possible_place + minPlace].frequency = 0;
                for(int k = 0; k < EMBEDDING_DIM; k++){
                    GPUEmbeddingAddress[possible_place + minPlace].a[k] = deviceMissingEmbedding[i].a[k];
                    GPUEmbeddingAddress[possible_place + minPlace].v[k] = deviceMissingEmbedding[i].v[k];
                }
                __threadfence();
                atomicExch(&locks[cache_id], 0);
                blocked = false;
            }
        }

    }
}

void CEmbeddingMap::InitEmbedding(std::string strFileloc, int bFirstLineDelete){
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
            tmp.key = nKeyTmp;
            tmp.a[i] = a_f;
            tmp.v[i] = v_f;
            tmp.frequency = 0;
        }
        EmbeddingOnDRAM.emplace_back(tmp);
        vKey.emplace_back(nKeyTmp);
    }

    totalMissCount = 0;
    totalHitCount = 0;
    totalBatch = 0;
    missingBatch = 0;

    //初始化CPU上的embedding map
    auto iter2 = EmbeddingOnDRAM.begin();
    for (auto iter1 = vKey.begin(); iter1 != vKey.end(); iter1++) {
        Set(*iter1,&(*iter2));
        iter2++;
    }

    //初始化组相联Cache的key为-1
    cudaMalloc((void **)&GPUEmbeddingAddress, CACHE_SIZE * sizeof(Parameters));
    InitEmptyCache<<<CACHE_SIZE / nDimBlock, nDimBlock>>>(GPUEmbeddingAddress);
    
    cudaMalloc((void**)&locks, CACHE_NUM * sizeof(int));
    cudaMemset(locks, 0, CACHE_NUM * sizeof(int));
    int length = EmbeddingOnDRAM.size();

    Parameters *AllGPUEmbeddings;
    cudaMalloc((void **)&AllGPUEmbeddings, length * sizeof(Parameters));
    cudaMemcpy(AllGPUEmbeddings, &EmbeddingOnDRAM[0], length * sizeof(Parameters), cudaMemcpyHostToDevice);

    DeviceInitEmbedding<<<length/nDimBlock + 1, nDimBlock>>>(locks, GPUEmbeddingAddress, AllGPUEmbeddings, length);

    ifDataSet.close();
}


void CEmbeddingMap::GatherBatch(const std::vector<int>& line, int cursor, Parameters *gatherResult, int currentBatchSize){ 
    //将Batch中的key拷贝到GPU
    int *keyBatch;
    cudaMalloc((void **)&keyBatch, currentBatchSize * sizeof(int));
    cudaMemcpy(keyBatch, &line[cursor], currentBatchSize * sizeof(int), cudaMemcpyHostToDevice);

    clock_gettime(CLOCK_MONOTONIC, &tStart);

    //创建查找到的embedding数据存储的空间
    Parameters *deviceGatherResult;
    cudaMalloc((void **)&deviceGatherResult, currentBatchSize * sizeof(Parameters));

    //创建MissList的空间
    int *devMissLock;
    cudaMalloc((void **)&devMissLock, sizeof(int));

    int *deviceMissKeyList, *missKeyList, *deviceMissIndexList;
    cudaMalloc((void **)&deviceMissKeyList, currentBatchSize * sizeof(int));
    cudaMalloc((void **)&deviceMissIndexList, currentBatchSize * sizeof(int));
    missKeyList = new int[currentBatchSize]();

    int missCount = 0, *devMissCount;
    cudaMalloc((void **)&devMissCount, sizeof(int));
    //Gather 
    GatherEmbedding<<<(BATCH_SIZE * EMBEDDING_DIM + nDimBlock - 1) / nDimBlock, nDimBlock>>>(keyBatch, GPUEmbeddingAddress, deviceGatherResult, devMissCount, deviceMissIndexList, deviceMissKeyList, devMissLock, currentBatchSize);
    
    cudaDeviceSynchronize();

    clock_gettime(CLOCK_MONOTONIC, &tEnd);
    hitTime += ((double)(tEnd.tv_sec - tStart.tv_sec)*1000000000 + tEnd.tv_nsec - tStart.tv_nsec)/1000000;

    //如果有缺少的，从CPU上拉取
    clock_gettime(CLOCK_MONOTONIC, &tStart);
    cudaMemcpy(&missCount, devMissCount, sizeof(int), cudaMemcpyDeviceToHost);
    totalBatch++;  
    clock_gettime(CLOCK_MONOTONIC, &tEnd);
    statusMemcpyTime += ((double)(tEnd.tv_sec - tStart.tv_sec)*1000000000 + tEnd.tv_nsec - tStart.tv_nsec)/1000000;

    if(missCount > 0){        
        clock_gettime(CLOCK_MONOTONIC, &tStart);
        missingBatch++;
        Parameters *missingEmbedding, *deviceMissingEmbedding;
        cudaMalloc(&deviceMissingEmbedding, missCount * sizeof(Parameters));
        missingEmbedding = new Parameters[missCount];
        
        
        cudaMemcpy(missKeyList, deviceMissKeyList, sizeof(int) * missCount, cudaMemcpyDeviceToHost);
        clock_gettime(CLOCK_MONOTONIC, &tEnd);
        memcpyTime += ((double)(tEnd.tv_sec - tStart.tv_sec)*1000000000 + tEnd.tv_nsec - tStart.tv_nsec)/1000000;

        clock_gettime(CLOCK_MONOTONIC, &tStart);
        //从CPU中查找缺失的Embedding
        //TODO::修改为多线程查找
        for(int i = 0; i < missCount; i++){
            Parameters *tmp;
            tmp = Get(missKeyList[i]);
            missingEmbedding[i].key = tmp->key;
            for(int j = 0;j < EMBEDDING_DIM; j++){
                missingEmbedding[i].a[j] = tmp->a[j];
                missingEmbedding[i].v[j] = tmp->v[j];
            }
            missingEmbedding[i].frequency = tmp->frequency;
        }
        clock_gettime(CLOCK_MONOTONIC, &tEnd);
        lookUpTime += ((double)(tEnd.tv_sec - tStart.tv_sec)*1000000000 + tEnd.tv_nsec - tStart.tv_nsec)/1000000;
        
        //将查询结果拷上GPU
        clock_gettime(CLOCK_MONOTONIC, &tStart);
        cudaMemcpy(deviceMissingEmbedding, missingEmbedding, missCount * sizeof(Parameters), cudaMemcpyHostToDevice);
        GatherMissingEmbedding<<<(missCount + nDimBlock - 1) / nDimBlock, nDimBlock>>>(locks, keyBatch, GPUEmbeddingAddress, deviceGatherResult, deviceMissIndexList, deviceMissKeyList, deviceMissingEmbedding, missCount);
        cudaDeviceSynchronize();
        clock_gettime(CLOCK_MONOTONIC, &tEnd);
        memcpyTime += ((double)(tEnd.tv_sec - tStart.tv_sec)*1000000000 + tEnd.tv_nsec - tStart.tv_nsec)/1000000;

        delete []missingEmbedding;
        cudaFree(deviceMissingEmbedding);
    }

    //将结果拷贝回CPU检验
    cudaMemcpy(&gatherResult[cursor], deviceGatherResult, currentBatchSize * sizeof(Parameters), cudaMemcpyDeviceToHost);


    totalHitCount += currentBatchSize - missCount;
    totalMissCount += missCount;


    delete []missKeyList;
    cudaFree(devMissLock);
    cudaFree(deviceMissKeyList);
    cudaFree(deviceMissIndexList);
    cudaFree(devMissCount);
    cudaFree(deviceGatherResult);
    cudaFree(keyBatch);
}

void CEmbeddingMap::GatherWork(const std::vector<int>& line, Parameters *gatherResult){
    int cursor = 0;
    int end = line.size();
    hitTime = 0;
    statusMemcpyTime = 0;
    lookUpTime = 0;
    memcpyTime = 0;

    while(end - cursor >= BATCH_SIZE){
        GatherBatch(line, cursor, gatherResult, BATCH_SIZE);
        cursor += BATCH_SIZE;
    }
    GatherBatch(line, cursor, gatherResult, end - cursor);
}

float CEmbeddingMap::GetHitRate(){
    return totalHitCount / (totalHitCount + totalMissCount);
}

float CEmbeddingMap::GetMissingBatchRate(){
    return missingBatch / totalBatch;
}

float CEmbeddingMap::GetHitTime(){
    return hitTime;
}

float CEmbeddingMap::GetStatusMemcpyTime(){
    return statusMemcpyTime;
}

float CEmbeddingMap::GetLookUpTime(){
    return lookUpTime;
}

float CEmbeddingMap::GetMemcpyTime(){
    return memcpyTime;
}

void CEmbeddingMap::MoveAllEmbeddings(Parameters *CPUEmbeddingAddress){
    cudaMemcpy(CPUEmbeddingAddress, GPUEmbeddingAddress, CACHE_SIZE * sizeof(Parameters), cudaMemcpyDeviceToHost);
}

void CEmbeddingMap::DeleteEmbedding(){
    cudaFree(locks);
    cudaFree(GPUEmbeddingAddress);
}

