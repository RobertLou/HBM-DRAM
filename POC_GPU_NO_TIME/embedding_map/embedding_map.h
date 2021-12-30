#pragma once
#include <iostream>
#include <unordered_map>
#include <fstream>
#include <vector>
#include <mutex>
#include <thread>
#include <sstream>
#include <math.h>
#include <time.h>


#include "cuda_runtime.h"
#include "device_launch_parameters.h"

const int nEvListSize = 8;
const int nBatchSize = 256 * 16 ;
const int nDimBlock = 256;

const int g = 1;
const int c = 1;

struct Parameters{
	float a[nEvListSize];	
	float v[nEvListSize]; 	//embedding
};

struct TimeInterval{
	timespec tMemStart, tMemEnd;
	float fMemcpyTime1 = 0,fMemcpyTime2 = 0;
};//ti用于记录每个线程中的各项任务的时间


__global__ void UpdateOneEmbedding(Parameters *batch);


class CEmbeddingMap{
private:
	std::mutex a_mutex;
	
public:
	std::unordered_map<int, Parameters *> a_map;
	Parameters* Get(int Key);

	void Set(int Key, Parameters* Value);

	void Erase(int key);

	void InitEmbedding(std::string fileloc,std::vector<Parameters> &lines,int firstlinedelete);

	void BatchWork(const std::vector<int>& line,int cursor,Parameters *batch,Parameters *batch_address,int current_batch_size,TimeInterval &ti);
	void Work(const std::vector<int>& line,int front,int end,int worker_id);
	void UpdateEV(const std::vector<int>& line);
};