#include "embedding_map.h"

Parameters* CEmbeddingMap::get(int Key){
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

void CEmbeddingMap::InitEmbedding(std::string strFileloc,std::vector<Parameters> &vLines,int bFirstLineDelete){
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
        vLines.emplace_back(tmp);
        vKey.emplace_back(nKeyTmp);
    }
    auto iter2 = vLines.begin();
    for (auto iter1 = vKey.begin(); iter1 != vKey.end(); iter1++) {
        Set(*iter1,&(*iter2));
        iter2++;
    }

    ifDataSet.close();
}

void CEmbeddingMap::UpdateBatch(const std::vector<int>& line,int cursor,Parameters *batch,int nCurrentBatchSize,TimeInterval &ti){
    Parameters* tmp;
    int nBatchCursor = 0;

    //memcpy,将查询到的数据复制到连续的batch空间中
    clock_gettime(CLOCK_MONOTONIC, &ti.tMemStart);
    for (auto iter = line.cbegin() + cursor; iter != line.cbegin() + cursor + nCurrentBatchSize; iter++) {
        tmp = get(*iter);
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
    for(int i = 0;i < nCurrentBatchSize;++i){
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
    for(auto iter = line.cbegin() + cursor; iter != line.cbegin() + cursor + nCurrentBatchSize; iter++) {
        tmp = get(*iter);
        for(int i = 0;i < EMBEDDING_DIM;++i){
            tmp->a[i] = batch[nBatchCursor].a[i] ;
            tmp->v[i] = batch[nBatchCursor].v[i] ;
        }

        nBatchCursor++;
    }
    clock_gettime(CLOCK_MONOTONIC, &ti.tMemEnd);
    ti.fMemcpyTime2 += ((double)(ti.tMemEnd.tv_sec - ti.tMemStart.tv_sec)*1000000000 + ti.tMemEnd.tv_nsec - ti.tMemStart.tv_nsec)/1000000;
}


void CEmbeddingMap::UpdateWork(const std::vector<int>& vLine,int nStart,int nEnd,int nWorkerId)
{	
    Parameters* tmp;
    int nCursor = nStart;
    int nBatchCursor = 0;
    Parameters *pBatch= new Parameters[BATCH_SIZE];
    TimeInterval ti;
    ti.fMemcpyTime1 = 0;
    ti.fMemcpyTime2 = 0;
    ti.fUpdateTime = 0;

    while(nEnd - nCursor >= BATCH_SIZE){
        UpdateBatch(vLine,nCursor,pBatch,BATCH_SIZE,ti);
        nCursor += BATCH_SIZE;
    }

    UpdateBatch(vLine,nCursor,pBatch,nEnd - nCursor,ti);

    delete []pBatch;

    std::cout << "线程" << nWorkerId << "已经结束" << std::endl;
    std::cout << "memcpy time 1:" << ti.fMemcpyTime1 << "ms" << std::endl;
    std::cout << "memcpy time 2:" << ti.fMemcpyTime2 << "ms" << std::endl;
    std::cout << "update time:" << ti.fUpdateTime << "ms" << std::endl;
}

void CEmbeddingMap::MultiThreadUpdateEV(const std::vector<int>& line) {
    int scope = line.size() / THREAD_NUM;
    std::thread th_arr[THREAD_NUM];

    for (unsigned int i = 0; i < THREAD_NUM - 1; ++i) {
        th_arr[i] = std::thread(&CEmbeddingMap::UpdateWork,this, std::ref(line), i * scope, (i + 1) * scope, i);
    }
    th_arr[THREAD_NUM - 1] = std::thread(&CEmbeddingMap::UpdateWork,this, std::ref(line), (THREAD_NUM - 1) * scope, line.size(), THREAD_NUM - 1);
    for (unsigned int i = 0; i < THREAD_NUM; ++i) {
        th_arr[i].join();
    }
}