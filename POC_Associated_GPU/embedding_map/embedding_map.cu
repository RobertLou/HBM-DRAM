#include "embedding_map.h"

__global__ void InitEmptyCache(Parameters *GPUEmbeddingAddress){
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    GPUEmbeddingAddress[i].key = -1;
}

__global__ void HostInitEmbedding(Parameters *GPUEmbeddingAddress, Parameters *AllGPUEmbeddings, int length){
	int i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i < length){
        int key = AllGPUEmbeddings[i].key;
        int cache_id = key / CACHE_NUM * WAYS;
        for(int j = 0;j < WAYS;j++){
            if(GPUEmbeddingAddress[cache_id + j].key == -1){
                GPUEmbeddingAddress[cache_id + j].key = i;
                for(int k = 0; k < EMBEDDING_DIM; k++){
                    GPUEmbeddingAddress[cache_id + j].a[k] = AllGPUEmbeddings[i].a[k];
                    GPUEmbeddingAddress[cache_id + j].v[k] = AllGPUEmbeddings[i].v[k];
                }
                break;
            }
        }
    }
}

__global__ void GatherEmbedding(int *keyBatch, Parameters *GPUEmbeddingAddress, Parameters *devicegatherResult, int currentBatchSize){
	int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j;
    if(i < currentBatchSize){
        int key = keyBatch[i];
        int cache_id = key / CACHE_NUM * WAYS;
        
        for(j = 0; j < WAYS; j++){
            if(GPUEmbeddingAddress[cache_id + j].key == key){
                devicegatherResult[i].key = key;
                for(int k = 0; k < EMBEDDING_DIM; k++){
                    devicegatherResult[i].a[k] = GPUEmbeddingAddress[cache_id + j].a[k];
                    devicegatherResult[i].v[k] = GPUEmbeddingAddress[cache_id + j].v[k];
                }
                break;
            }
        }
        if(j == WAYS){
            for(int k = 0; k < EMBEDDING_DIM; k++){
                devicegatherResult[i].a[k] = -1;
                devicegatherResult[i].v[k] = -1;
            }
        }
    }
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
            tmp.key = nKeyTmp;
            tmp.a[i] = a_f;
            tmp.v[i] = v_f;
            tmp.frequency = 0;
        }
        line.emplace_back(tmp);
        vKey.emplace_back(nKeyTmp);
    }
    //初始化组相联Cache的key为-1
    cudaMalloc((void **)&GPUEmbeddingAddress, CACHE_SIZE * sizeof(Parameters));
    InitEmptyCache<<<CACHE_SIZE / nDimBlock, nDimBlock>>>(GPUEmbeddingAddress);

    int length = line.size();

    Parameters *AllGPUEmbeddings;
    cudaMalloc((void **)&AllGPUEmbeddings, length * sizeof(Parameters));
    cudaMemcpy(AllGPUEmbeddings, &line[0], length * sizeof(Parameters), cudaMemcpyHostToDevice);

    HostInitEmbedding<<<length/nDimBlock + 1, nDimBlock>>>(GPUEmbeddingAddress, AllGPUEmbeddings, length);

    ifDataSet.close();
}


void CEmbeddingMap::GatherBatch(const std::vector<int>& line, int cursor, Parameters *gatherResult, int currentBatchSize){ 


    //将Batch中的key拷贝到GPU
    int *keyBatch;
    cudaMalloc((void **)&keyBatch, currentBatchSize * sizeof(int));
    cudaMemcpy(keyBatch, &line[cursor], currentBatchSize * sizeof(int), cudaMemcpyHostToDevice);

    //创建查找到的embedding数据存储的空间
    Parameters *devicegatherResult;
    cudaMalloc((void **)&devicegatherResult, currentBatchSize * sizeof(Parameters));

    //Gather 
    GatherEmbedding<<<BATCH_SIZE/nDimBlock, nDimBlock>>>(keyBatch, GPUEmbeddingAddress, devicegatherResult, currentBatchSize);
    cudaDeviceSynchronize();

    //将结果拷贝回CPU检验
    cudaMemcpy(&gatherResult[cursor], devicegatherResult, currentBatchSize * sizeof(Parameters), cudaMemcpyDeviceToHost);
    
    cudaFree(devicegatherResult);
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

void CEmbeddingMap::DeleteEmbedding(){
    cudaFree(GPUEmbeddingAddress);
}