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
#define CACHE_NUM (CACHE_SIZE / WAYS)

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
__global__ void DeviceInitEmbedding(int *locks, Parameters *GPUEmbeddingAddress, Parameters *AllGPUEmbeddings, int length);
__global__ void GatherEmbedding(int *locks, int *keyBatch, Parameters *GPUEmbeddingAddress, Parameters *deviceGatherResult, int *deviceGatherStatus, int devMissCount, int currentBatchSize);

class CEmbeddingMap{
private:
	std::unordered_map<int, Parameters *> a_map; 	//EmbeddingMap on DRAM
	std::mutex a_mutex;
	std::vector<Parameters> EmbeddingOnDRAM;		//Embedding storing place on DRAM
	Parameters *GPUEmbeddingAddress;				//Embedding storing place on GPU
	int *locks;			
	float totalMissCount, totalHitCount, totalBatch, missingBatch;							

public:
	Parameters* Get(int Key);
	void Set(int Key, Parameters* Value);
	void Erase(int key);
	void InitEmbedding(std::string strFileloc, int bFirstLineDelete);


	void GatherBatch(const std::vector<int>& line, int cursor, Parameters *batch, int currentBatchSize);
	void GatherWork(const std::vector<int>& line, Parameters *gatherResult);
	
	float GetHitRate();
	float GetMissingBatchRate();
	
	//Move all embeddings from GPU cache to memory 
	void MoveAllEmbeddings(Parameters *CPUEmbeddingAddress);

	void DeleteEmbedding();
};