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

#define THREAD_NUM 4
#define EMBEDDING_DIM 128
#define BATCH_SIZE 256 * 16

const int nDimBlock = 256;

const int g = 1;
const int c = 1;

struct Parameters{
	float a[EMBEDDING_DIM];	
	float v[EMBEDDING_DIM]; 	//embedding
};

struct TimeInterval{
	timespec tMemStart, tMemEnd;
	float fMemcpyTime1 = 0,fMemcpyTime2 = 0;
};//ti用于记录每个线程中的各项任务的时间


__global__ void UpdateOneEmbedding(Parameters *batch, int currentBatchSize);
__global__ void GatherEmbedding(Parameters **deviceAddressBatch, Parameters *devicegatherResult, int currentBatchSize);

class CEmbeddingMap{
private:
	std::mutex a_mutex;
	Parameters *GPUEmbeddingAddress;

public:
	std::unordered_map<int, Parameters *> a_map;
	
	Parameters* Get(int Key);
	void Set(int Key, Parameters* Value);
	void Erase(int key);

	void InitEmbedding(std::string strFileloc, std::vector<Parameters> &line, int bFirstLineDelete);

	void UpdateBatch(const std::vector<int>& line, int nCursor, Parameters **addressBatch, int nCurrentBatchSize, TimeInterval &ti);
	void UpdateWork(const std::vector<int>& line, int start, int end, int workerId);
	void MultiThreadUpdateEV(const std::vector<int>& line);

	void GatherBatch(const std::vector<int>& line, int cursor, Parameters *batch, int currentBatchSize);
	void GatherWork(const std::vector<int>& line, Parameters *gatherResult, int start, int end, int worker_id);
	void MultiThreadGatherEV(const std::vector<int>& line, Parameters *gatherResult);

	void DeleteEmbedding();
};