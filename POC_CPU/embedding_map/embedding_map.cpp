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

void CEmbeddingMap::UpdateBatch(const std::vector<int>& line,int cursor,Parameters *batch,int currentBatchSize,TimeInterval &ti){
    Parameters* tmp;
    int nBatchCursor = 0;

    //memcpy,将查询到的数据复制到连续的batch空间中
    clock_gettime(CLOCK_MONOTONIC, &ti.tMemStart);
    for (auto iter = line.cbegin() + cursor; iter != line.cbegin() + cursor + currentBatchSize; iter++) {
        tmp = Get(*iter);
        for(int i = 0;i < EMBEDDING_DIM;++i){
            batch[nBatchCursor].a[i] = tmp->a[i];
            batch[nBatchCursor].v[i] = tmp->v[i];
        }
        nBatchCursor++;
    }
    clock_gettime(CLOCK_MONOTONIC, &ti.tMemEnd);
    ti.fMemcpyTime1 += ((double)(ti.tMemEnd.tv_sec - ti.tMemStart.tv_sec)*1000000000 + ti.tMemEnd.tv_nsec - ti.tMemStart.tv_nsec)/1000000;


    //计算更新embedding
    clock_gettime(CLOCK_MONOTONIC, &ti.tMemStart);
    for(int i = 0;i < currentBatchSize;++i){
        for(int j = 0;j < EMBEDDING_DIM;j++){
            batch[i].a[j] += g * g;
            batch[i].v[j] -= (c * g * 1.0) / sqrt(batch[i].a[j]);
        }
    }
    clock_gettime(CLOCK_MONOTONIC, &ti.tMemEnd);
    ti.fUpdateTime += ((double)(ti.tMemEnd.tv_sec - ti.tMemStart.tv_sec)*1000000000 + ti.tMemEnd.tv_nsec - ti.tMemStart.tv_nsec)/1000000;

    //memcpy，将更新后的数据拷回
    nBatchCursor = 0;
    clock_gettime(CLOCK_MONOTONIC, &ti.tMemStart);
    for(auto iter = line.cbegin() + cursor; iter != line.cbegin() + cursor + currentBatchSize; iter++) {
        tmp = Get(*iter);
        for(int i = 0;i < EMBEDDING_DIM;++i){
            tmp->a[i] = batch[nBatchCursor].a[i] ;
            tmp->v[i] = batch[nBatchCursor].v[i] ;
        }

        nBatchCursor++;
    }
    clock_gettime(CLOCK_MONOTONIC, &ti.tMemEnd);
    ti.fMemcpyTime2 += ((double)(ti.tMemEnd.tv_sec - ti.tMemStart.tv_sec)*1000000000 + ti.tMemEnd.tv_nsec - ti.tMemStart.tv_nsec)/1000000;
}


void CEmbeddingMap::UpdateWork(const std::vector<int>& line, int start, int end, int workerId)
{	
    Parameters* tmp;
    int cursor = start;
    Parameters *batch= new Parameters[BATCH_SIZE];
    TimeInterval ti;
    ti.fMemcpyTime1 = 0;
    ti.fMemcpyTime2 = 0;
    ti.fUpdateTime = 0;

    while(end - cursor >= BATCH_SIZE){
        UpdateBatch(line, cursor, batch, BATCH_SIZE, ti);
        cursor += BATCH_SIZE;
    }

    UpdateBatch(line, cursor, batch, end - cursor, ti);

    delete []batch;

    std::cout << "线程" << workerId << "已经结束" << std::endl;
    std::cout << "memcpy time 1:" << ti.fMemcpyTime1 << "ms" << std::endl;
    std::cout << "memcpy time 2:" << ti.fMemcpyTime2 << "ms" << std::endl;
    std::cout << "update time:" << ti.fUpdateTime << "ms" << std::endl;
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

void CEmbeddingMap::GatherBatch(const std::vector<int>& line, int cursor, Parameters *gatherResult, int currentBatchSize, TimeInterval &ti){ 
    Parameters* tmp;
    int nBatchCursor = 0;

    //memcpy,将查询到的数据复制到连续的batch空间中
    for (auto iter = line.cbegin() + cursor; iter != line.cbegin() + cursor + currentBatchSize; iter++) {
        clock_gettime(CLOCK_MONOTONIC, &ti.tMemStart);
        tmp = Get(*iter);
        clock_gettime(CLOCK_MONOTONIC, &ti.tMemEnd);
        ti.gatherLookupTime += ((double)(ti.tMemEnd.tv_sec - ti.tMemStart.tv_sec)*1000000000 + ti.tMemEnd.tv_nsec - ti.tMemStart.tv_nsec)/1000000;

        clock_gettime(CLOCK_MONOTONIC, &ti.tMemStart);
        for(int i = 0;i < EMBEDDING_DIM;++i){
            gatherResult[cursor + nBatchCursor].a[i] = tmp->a[i];
            gatherResult[cursor + nBatchCursor].v[i] = tmp->v[i];
        }
        nBatchCursor++;
        clock_gettime(CLOCK_MONOTONIC, &ti.tMemEnd);
        ti.gatherMemcpyTime += ((double)(ti.tMemEnd.tv_sec - ti.tMemStart.tv_sec)*1000000000 + ti.tMemEnd.tv_nsec - ti.tMemStart.tv_nsec)/1000000;
    }
}

void CEmbeddingMap::GatherWork(const std::vector<int>& line, Parameters *gatherResult, int start, int end, int workerId, TimeInterval &ti){
    int cursor = start;
    ti.gatherLookupTime = 0;
    ti.gatherMemcpyTime = 0;
    while(end - cursor >= BATCH_SIZE){
        GatherBatch(line, cursor, gatherResult, BATCH_SIZE, ti);
        cursor += BATCH_SIZE;
    }
    GatherBatch(line, cursor, gatherResult, end - cursor, ti);
}

void CEmbeddingMap::MultiThreadGatherEV(const std::vector<int>& line, Parameters *gatherResult) {
    int scope = line.size() / THREAD_NUM;
    std::thread th_arr[THREAD_NUM];
    TimeInterval ti[THREAD_NUM];
    for (unsigned int i = 0; i < THREAD_NUM - 1; ++i) {
        th_arr[i] = std::thread(&CEmbeddingMap::GatherWork, this, std::ref(line), gatherResult, i * scope, (i + 1) * scope, i, std::ref(ti[i]));
    }
    th_arr[THREAD_NUM - 1] = std::thread(&CEmbeddingMap::GatherWork, this, std::ref(line), gatherResult, (THREAD_NUM - 1) * scope, line.size(), THREAD_NUM - 1, std::ref(ti[THREAD_NUM - 1]));
    for (unsigned int i = 0; i < THREAD_NUM; ++i) {
        th_arr[i].join();
    }

    float averageLookupTime = 0,averageMemcpyTime = 0;
    for (unsigned int i = 0; i < THREAD_NUM; ++i) {
        averageLookupTime += ti[i].gatherLookupTime;
        averageMemcpyTime += ti[i].gatherMemcpyTime;
    }

    std::cout << "LookupTime:" << averageLookupTime / THREAD_NUM << "ms" << std::endl;
    std::cout << "MemcpyTime:" << averageMemcpyTime / THREAD_NUM << "ms" << std::endl;
}