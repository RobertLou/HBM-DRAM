#pragma once
#include <iostream>
#include <unordered_map>
#include <fstream>
#include <vector>
#include <mutex>
#include <shared_mutex>
#include <thread>
#include <sstream>
#include <math.h>
#include <time.h>


#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cooperative_groups.h>

#define THREAD_NUM 4
#define EMBEDDING_DIM 128
#define BATCH_SIZE 2048
#define BLOCK_SIZE 128
#define CACHE_SIZE 65536
#define WAYS 8
#define CACHE_NUM (CACHE_SIZE / WAYS)

// slab for static slab list
#define WARP_SIZE 32
#define SET_ASSOCIATIVITY 2

namespace cg = cooperative_groups;

const int nDimBlock = 256;

const int g = 1;
const int c = 1;

struct Parameters{
	int key;
	float a[EMBEDDING_DIM];	
	float v[EMBEDDING_DIM]; 	//embedding
	int frequency;
};

struct static_slab {
  int slab_[WARP_SIZE];
};

// Static slablist(slabset) for GPU Cache
struct slab_set {
  static_slab set_[SET_ASSOCIATIVITY];
};

struct TimeInterval{
	timespec tMemStart, tMemEnd;
	float fMemcpyTime1 = 0,fMemcpyTime2 = 0;
};//ti用于记录每个线程中的各项任务的时间

__global__ void InitEmptyCache(Parameters *GPUEmbeddingAddress);
__global__ void DeviceInitEmbedding(int *locks, Parameters *GPUEmbeddingAddress, Parameters *AllGPUEmbeddings, int length);
__global__ void GatherEmbedding(int *locks, int *keyBatch, Parameters *GPUEmbeddingAddress, Parameters *deviceGatherResult, int *deviceGatherStatus, int devMissCount, int currentBatchSize);
__global__ void GatherMissingEmbedding(int *locks, int *keyBatch, Parameters *GPUEmbeddingAddress, Parameters *deviceGatherResult, int *deviceGatherStatus, Parameters *deviceMissingEmbedding, int currentBatchSize);

class CEmbeddingMap{
private:
	std::unordered_map<int, Parameters *> a_map; 	//EmbeddingMap on DRAM
	std::shared_mutex a_mutex;
	std::vector<Parameters> EmbeddingOnDRAM;		//Embedding storing place on DRAM
	
	int* set_mutex_ = nullptr;
	slab_set* keys_ = nullptr;
  	float* vals_ = nullptr;
  	int* slot_counter_ = nullptr;
  	int* global_counter_ = nullptr;					//Embedding storing place on GPU

	// Cache capacity
	int capacity_in_set_;
	int num_slot_;

	// Embedding vector size
	int embedding_vec_size_;

	float totalMissCount, totalHitCount, totalBatch, missingBatch;		
	timespec tStart, tEnd;					
	float hitTime, statusMemcpyTime, lookUpTime, memcpyTime;
public:
	Parameters* Get(int Key);
	void Set(int Key, Parameters* Value);
	void Erase(int key);
	void InitEmbedding(std::string strFileloc, int bFirstLineDelete);


	void GatherBatch(const std::vector<int>& line, int cursor, Parameters *batch, int currentBatchSize);
	void GatherWork(const std::vector<int>& line, Parameters *gatherResult);
	
	float GetHitRate();
	float GetMissingBatchRate();
	float GetHitTime();
	float GetStatusMemcpyTime();
	float GetLookUpTime();
	float GetMemcpyTime();
	
	//Move all embeddings from GPU cache to memory 
	void MoveAllEmbeddings(Parameters *CPUEmbeddingAddress);

	void DeleteEmbedding();
};