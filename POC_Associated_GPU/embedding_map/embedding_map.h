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
#define CACHE_SIZE 262144
#define WAYS 8
#define CACHE_NUM CACHE_SIZE / WAYS

const int nDimBlock = 256;

const int g = 1;
const int c = 1;

struct Parameters{
	int key;
	float a[EMBEDDING_DIM];	
	float v[EMBEDDING_DIM]; 	//embedding
	int frequency;
};

struct TimeInterval{
	timespec tMemStart, tMemEnd;
	float fMemcpyTime1 = 0,fMemcpyTime2 = 0;
};//ti用于记录每个线程中的各项任务的时间

__global__ void InitEmptyCache(Parameters *GPUEmbeddingAddress);
__global__ void UpdateOneEmbedding(Parameters *batch, int currentBatchSize);
__global__ void InitEmbedding(Parameters *GPUEmbeddingAddress, Parameters *AllGPUEmbeddings, int length);
__global__ void GatherEmbedding(Parameters **deviceAddressBatch, Parameters *devicegatherResult, int currentBatchSize);

class CEmbeddingMap{
private:
	Parameters *GPUEmbeddingAddress;

public:

	void InitEmbedding(std::string strFileloc, std::vector<Parameters> &line, int bFirstLineDelete);


	void GatherBatch(const std::vector<int>& line, int cursor, Parameters *batch, int currentBatchSize);
	void GatherWork(const std::vector<int>& line, Parameters *gatherResult);

	void DeleteEmbedding();
};