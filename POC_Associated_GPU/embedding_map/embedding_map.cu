#include "embedding_map.h"

Parameters* CEmbeddingMap::Get(int Key){
	std::lock_guard<std::mutex> guard(a_mutex);
	return a_map.at(Key);
};

void CEmbeddingMap::Set(int Key, Parameters* Value){
	std::lock_guard<std::mutex> guard(a_mutex);
	a_map.insert(std::make_pair(Key, Value)); 
};

void CEmbeddingMap::Erase(int key){
	std::lock_guard<std::mutex> guard(a_mutex);
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

__global__ void GatherEmbedding(int *keyBatch, Parameters *GPUEmbeddingAddress, Parameters *deviceGatherResult, int *deviceGatherStatus, int *devMissCount, int currentBatchSize){
	int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j;
    if(i < currentBatchSize){
        int key = keyBatch[i];
        int cache_id = key % CACHE_NUM;
        int possible_place = cache_id * WAYS;
        deviceGatherStatus[i] = 0;
        for(j = 0; j < WAYS; j++){
            if(GPUEmbeddingAddress[possible_place + j].key == key){
                deviceGatherResult[i].key = key;
                deviceGatherStatus[i] = j + 1;
                for(int k = 0; k < EMBEDDING_DIM; k++){
                    deviceGatherResult[i].a[k] = GPUEmbeddingAddress[possible_place + j].a[k];
                    deviceGatherResult[i].v[k] = GPUEmbeddingAddress[possible_place + j].v[k];
                }
                atomicAdd(&GPUEmbeddingAddress[possible_place + j].frequency, 1);
                break;
            }
            
            if(-1 == atomicCAS(&GPUEmbeddingAddress[possible_place + j].key, -1, key)){
                atomicAdd(devMissCount, 1);
                deviceGatherStatus[i] = -j - 1;
                break;
            }
        }
        if(j == WAYS){
            atomicAdd(devMissCount, 1);
        }
    }
}

__global__ void GatherMissingEmbedding(int *locks, int *keyBatch, Parameters *GPUEmbeddingAddress, Parameters *deviceGatherResult, int *deviceGatherStatus, Parameters *deviceMissingEmbedding, int currentBatchSize){
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i < currentBatchSize){
        int key = keyBatch[i];
        int cache_id = key % CACHE_NUM;
        int possible_place = cache_id * WAYS;
        if(deviceGatherStatus[i] < 0){
            //写入Result
            deviceGatherResult[i].key =  key;
            for(int k = 0; k < EMBEDDING_DIM; k++){
                deviceGatherResult[i].a[k] = deviceMissingEmbedding[i].a[k];
                deviceGatherResult[i].v[k] = deviceMissingEmbedding[i].v[k];
            }
            deviceGatherResult[i].frequency = 0;

            //更新Cache
            int offset = - deviceGatherStatus[i] - 1;
            GPUEmbeddingAddress[possible_place + offset].key = key;
            for(int k = 0; k < EMBEDDING_DIM; k++){
                GPUEmbeddingAddress[possible_place + offset].a[k] = deviceMissingEmbedding[i].a[k];
                GPUEmbeddingAddress[possible_place + offset].v[k] = deviceMissingEmbedding[i].v[k];
            }
            GPUEmbeddingAddress[possible_place + offset].frequency = 0;
        }
        else if(deviceGatherStatus[i] == 0){
            //写入Result
            deviceGatherResult[i].key =  key;
            for(int k = 0; k < EMBEDDING_DIM; k++){
                deviceGatherResult[i].a[k] = deviceMissingEmbedding[i].a[k];
                deviceGatherResult[i].v[k] = deviceMissingEmbedding[i].v[k];
            }
            deviceGatherResult[i].frequency = 0;

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
                    for(int k = 0; k < EMBEDDING_DIM; k++){
                        GPUEmbeddingAddress[possible_place + minPlace].a[k] = deviceMissingEmbedding[i].a[k];
                        GPUEmbeddingAddress[possible_place + minPlace].v[k] = deviceMissingEmbedding[i].v[k];
                    }
                    GPUEmbeddingAddress[possible_place + minPlace].frequency = 0;

                    atomicExch(&locks[cache_id], 0);
                    blocked = false;
                }
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

    //创建查找到的embedding数据存储的空间
    Parameters *deviceGatherResult;
    cudaMalloc((void **)&deviceGatherResult, currentBatchSize * sizeof(Parameters));

    //创建Status的空间
    int *deviceGatherStatus,*gatherStatus;
    cudaMalloc((void **)&deviceGatherStatus, currentBatchSize * sizeof(int));
    gatherStatus = new int[currentBatchSize]();

    int missCount = 0, *devMissCount;
    cudaMalloc((void **)&devMissCount, sizeof(int));

    //Gather 
    GatherEmbedding<<<BATCH_SIZE/nDimBlock, nDimBlock>>>(keyBatch, GPUEmbeddingAddress, deviceGatherResult, deviceGatherStatus, devMissCount, currentBatchSize);
    cudaDeviceSynchronize();

    //如果有缺少的，从CPU上拉取
    cudaMemcpy(gatherStatus, deviceGatherStatus, currentBatchSize * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&missCount, devMissCount, sizeof(int), cudaMemcpyDeviceToHost);
    if(missCount > 0){
        Parameters *missingEmbedding, *deviceMissingEmbedding;
        cudaMalloc(&deviceMissingEmbedding, currentBatchSize * sizeof(Parameters));
        missingEmbedding = new Parameters[currentBatchSize];        

        //从CPU中查找缺失的Embedding
        //TODO::修改为多线程查找
        for(int i = 0; i < currentBatchSize; i++){
            if(gatherStatus[i] <= 0){
                Parameters *tmp;
                tmp = Get(line[cursor + i]);
                missingEmbedding[i].key = tmp->key;
                for(int j = 0;j < EMBEDDING_DIM; j++){
                    missingEmbedding[i].a[j] = tmp->a[j];
                    missingEmbedding[i].v[j] = tmp->v[j];
                }
                missingEmbedding[i].frequency = tmp->frequency;
            }
        }

        //将查询结果拷上GPU
        cudaMemcpy(deviceMissingEmbedding, missingEmbedding, currentBatchSize * sizeof(Parameters), cudaMemcpyHostToDevice);
        GatherMissingEmbedding<<<BATCH_SIZE/nDimBlock, nDimBlock>>>(locks, keyBatch, GPUEmbeddingAddress, deviceGatherResult, deviceGatherStatus, deviceMissingEmbedding, currentBatchSize);


        delete []missingEmbedding;
        cudaFree(deviceMissingEmbedding);
    }

    //将结果拷贝回CPU检验
    cudaMemcpy(&gatherResult[cursor], deviceGatherResult, currentBatchSize * sizeof(Parameters), cudaMemcpyDeviceToHost);

    
    int replaceCount = 0, missCount2 = 0, hitCount = 0;
    for(int i = 0; i < currentBatchSize;i++){
        if(gatherStatus[i] == 0){
            replaceCount++;
        }
        else if(gatherStatus[i] < 0){
            missCount2++;
        }
        else if(gatherStatus[i] > 0){
            hitCount++;
        }
    }
    //std::cout << missCount << std::endl;
    //std::cout << hitCount << "," << replaceCount << "," << missCount2 << std::endl;



    delete []gatherStatus;
    cudaFree(deviceGatherStatus);
    cudaFree(deviceGatherResult);
    cudaFree(keyBatch);
}

void CEmbeddingMap::GatherWork(const std::vector<int>& line, Parameters *gatherResult){
    int cursor = 0;
    int end = line.size();

    while(end - cursor >= BATCH_SIZE){
        GatherBatch(line, cursor, gatherResult, BATCH_SIZE);
        cursor += BATCH_SIZE;
    }
    GatherBatch(line, cursor, gatherResult, end - cursor);
}

void CEmbeddingMap::MoveAllEmbeddings(Parameters *CPUEmbeddingAddress){
    cudaMemcpy(CPUEmbeddingAddress, GPUEmbeddingAddress, CACHE_SIZE * sizeof(Parameters), cudaMemcpyDeviceToHost);
}

void CEmbeddingMap::DeleteEmbedding(){
    cudaFree(locks);
    cudaFree(GPUEmbeddingAddress);
}