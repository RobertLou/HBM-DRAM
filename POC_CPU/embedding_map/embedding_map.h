#pragma once
#include <iostream>
#include <unordered_map>
#include <fstream>
#include <vector>
#include <mutex>
#include <thread>
#include <math.h>
#include <time.h>
#include <sstream> 

#define THREAD_NUM 4
#define EMBEDDING_DIM 8
#define BATCH_SIZE 256 * 16

struct Parameters{
	float a[EMBEDDING_DIM];	
	float v[EMBEDDING_DIM]; 	//embedding
};

struct TimeInterval{
	timespec tMemStart, tMemEnd;
	float fMemcpyTime1,fMemcpyTime2,fUpdateTime;
};//ti用于记录每个线程中的各项任务的时间

class CEmbeddingMap{
private:
	std::mutex a_mutex;
	const int g = 1;
	const int c = 1;
public:
	std::unordered_map<int, Parameters *> a_map;

	Parameters* get(int Key);
	void Set(int Key, Parameters* Value);
	void Erase(int key);
	void InitEmbedding(std::string strFileloc,std::vector<Parameters> &lines,int firstlinedelete);

	void UpdateBatch(const std::vector<int>& line,int cursor,Parameters *batch,int current_batch_size,TimeInterval &ti);
	void UpdateWork(const std::vector<int>& line,int start,int end,int worker_id);
	void MultiThreadUpdateEV(const std::vector<int>& line);
};