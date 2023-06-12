#include "embedding_map.h"

__global__ void UpdateOneEmbedding(Parameters *Batch){
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	for(int j = 0;j < EMBEDDING_DIM;j++){
        Batch[i].a[j] += g * g;
        Batch[i].v[j] -= (c * g * 1.0) / sqrt(Batch[i].a[j]);
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
    
    auto iter2 = line.begin();
    for (auto iter1 = vKey.begin(); iter1 != vKey.end(); iter1++) {
        Set(*iter1,&(*iter2));
        iter2++;
    }

    ifDataSet.close();
}

void CEmbeddingMap::UpdateBatch(const std::vector<int>& line, int nCursor, Parameters *Batch, Parameters *BatchAddressGPU, int nCurrentBatchSize, TimeInterval &ti){
    Parameters* tmp;  
    int nBatchCursor = 0;

    //memcpy,将查询到的数据复制到连续的Batch空间中
    clock_gettime(CLOCK_MONOTONIC, &ti.tMemStart);
    for (auto iter = line.cbegin() + nCursor; iter != line.cbegin() + nCursor + nCurrentBatchSize; iter++) {
        tmp = Get(*iter);
        for(int i = 0;i < EMBEDDING_DIM;++i){
            Batch[nBatchCursor].a[i] = tmp->a[i];
            Batch[nBatchCursor].v[i] = tmp->v[i];
        }
        nBatchCursor++;
    }
    clock_gettime(CLOCK_MONOTONIC, &ti.tMemEnd);
    ti.fMemcpyTime1 += ((double)(ti.tMemEnd.tv_sec - ti.tMemStart.tv_sec)*1000000000 + ti.tMemEnd.tv_nsec - ti.tMemStart.tv_nsec)/1000000;


    //计算更新embedding
    cudaMemcpy(BatchAddressGPU, Batch, nCurrentBatchSize * sizeof(Parameters), cudaMemcpyHostToDevice);
    UpdateOneEmbedding<<<BATCH_SIZE/nDimBlock,nDimBlock>>>(BatchAddressGPU);
    cudaMemcpy(Batch, BatchAddressGPU, nCurrentBatchSize * sizeof(Parameters), cudaMemcpyDeviceToHost);
        
    //memcpy，将更新后的数据拷回
    nBatchCursor = 0;
    clock_gettime(CLOCK_MONOTONIC, &ti.tMemStart);
    for(auto iter = line.cbegin() + nCursor; iter != line.cbegin() + nCursor + nCurrentBatchSize; iter++) {
        tmp = Get(*iter);
        for(int i = 0;i < EMBEDDING_DIM;++i){
            tmp->a[i] = Batch[nBatchCursor].a[i] ;
            tmp->v[i] = Batch[nBatchCursor].v[i] ;
        }
        nBatchCursor++;
    }
    clock_gettime(CLOCK_MONOTONIC, &ti.tMemEnd);
    ti.fMemcpyTime2 += ((double)(ti.tMemEnd.tv_sec - ti.tMemStart.tv_sec)*1000000000 + ti.tMemEnd.tv_nsec - ti.	tMemStart.tv_nsec)/1000000;
}

void CEmbeddingMap::UpdateWork(const std::vector<int>& line, int start, int end, int workerId)
	{	
		int cursor = start;
		Parameters *Batch= new Parameters[BATCH_SIZE];

		Parameters *BatchAddressGPU;
		TimeInterval ti;

		cudaMalloc((void **)&BatchAddressGPU, BATCH_SIZE * sizeof(Parameters));
		while(end - cursor >= BATCH_SIZE){
			UpdateBatch(line, cursor, Batch, BatchAddressGPU, BATCH_SIZE, ti);
			cursor += BATCH_SIZE;
		}
		UpdateBatch(line, cursor, Batch, BatchAddressGPU, end - cursor, ti);
		delete []Batch;
		cudaFree(BatchAddressGPU);

		std::cout << "线程" << workerId << "已经结束" << std::endl;
		std::cout << "memcpy time 1:" << ti.fMemcpyTime1 << "ms" << std::endl;		//CPU memcpy time
		std::cout << "memcpy time 2:" << ti.fMemcpyTime2 << "ms" << std::endl;
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