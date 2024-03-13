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

__device__ void warp_tile_copy(const int lane_idx,
                                               const int emb_vec_size_in_float, float* d_dst,
                                               const float* d_src) {
#pragma unroll
  for (int i = lane_idx; i < emb_vec_size_in_float; i += WARP_SIZE) {
    d_dst[i] = d_src[i];
  }
}

// Will be called by multiple thread_block_tile((sub-)warp) on the same mutex
// Expect only one thread_block_tile return to execute critical section at any time
__forceinline__ __device__ void warp_lock_mutex(const cg::thread_block_tile<WARP_SIZE>& warp_tile,
                                                 int& set_mutex) {
  // The first thread of this (sub-)warp to acquire the lock
  if (warp_tile.thread_rank() == 0) {
    while (0 == atomicCAS((int*)&set_mutex, 1, 0))
      ;
  }
  __threadfence();
  warp_tile.sync();  // Synchronize the threads in the (sub-)warp. Execution barrier + memory fence
}

// The (sub-)warp holding the mutex will unlock the mutex after finishing the critical section on a
// set Expect any following (sub-)warp that acquire the mutex can see its modification done in the
// critical section
__forceinline__ __device__ void warp_unlock_mutex(const cg::thread_block_tile<WARP_SIZE>& warp_tile,
                                                   int& set_mutex) {
  __threadfence();
  warp_tile.sync();  // Synchronize the threads in the (sub-)warp. Execution barrier + memory fence
  // The first thread of this (sub-)warp to release the lock
  if (warp_tile.thread_rank() == 0) {
    atomicExch((int*)&set_mutex, 1);
  }
}

// The (sub-)warp doing all reduction to find the slot with min slot_counter
// The slot with min slot_counter is the LR slot.
__forceinline__ __device__ void warp_min_reduction(
    const cg::thread_block_tile<WARP_SIZE>& warp_tile, int& min_slot_counter_val,
    int& slab_distance, int& slot_distance) {
  const int lane_idx = warp_tile.thread_rank();
  slot_distance = lane_idx;

  for (int i = (warp_tile.size() >> 1); i > 0; i = i >> 1) {
    int input_slot_counter_val = warp_tile.shfl_xor(min_slot_counter_val, (int)i);
    int input_slab_distance = warp_tile.shfl_xor(slab_distance, (int)i);
    int input_slot_distance = warp_tile.shfl_xor(slot_distance, (int)i);

    if (input_slot_counter_val == min_slot_counter_val) {
      if (input_slab_distance == slab_distance) {
        if (input_slot_distance < slot_distance) {
          slot_distance = input_slot_distance;
        }
      } else if (input_slab_distance < slab_distance) {
        slab_distance = input_slab_distance;
        slot_distance = input_slot_distance;
      }
    } else if (input_slot_counter_val < min_slot_counter_val) {
      min_slot_counter_val = input_slot_counter_val;
      slab_distance = input_slab_distance;
      slot_distance = input_slot_distance;
    }
  }
}

__global__ void update_kernel_overflow_ignore(int* global_counter,
                                              int* d_missing_len) {
  // Update global counter
  atomicAdd(global_counter, 1);
  *d_missing_len = 0;
}

__global__ void init_cache(slab_set* keys, int* slot_counter,
                           int* global_counter, const int num_slot,
                           const int empty_key, int* set_mutex, const int capacity_in_set) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_slot) {
    // Set the key of this slot to unused key
    // Flatten the cache
    int* key_slot = (int*)keys;
    key_slot[idx] = empty_key;

    // Clear the counter for this slot
    slot_counter[idx] = 0;
  }
  // First CUDA thread clear the global counter
  if (idx == 0) {
    global_counter[idx] = 0;
  }

  // First capacity_in_set CUDA thread initialize mutex
  if (idx < capacity_in_set) {
    set_mutex[idx] = 1;
  }
}

// Kernel to read from cache
// Also update locality information for touched slot
__global__ void get_kernel(const int* d_keys, const int len, float* d_values,
                           const int embedding_vec_size, int* d_missing_index,
                           int* d_missing_keys, int* d_missing_len, int* miss_count,
                           int* global_counter,
                           int* slot_counter, const int capacity_in_set,
                           slab_set* keys, float* vals, int* set_mutex,
                           const int task_per_warp_tile) {
  int empty_key = -1;
  // Lane(thread) ID within a warp_tile
  cg::thread_block_tile<WARP_SIZE> warp_tile =
      cg::tiled_partition<WARP_SIZE>(cg::this_thread_block());
  const int lane_idx = warp_tile.thread_rank();
  // Warp tile global ID
  const int warp_tile_global_idx =
      (blockIdx.x * (blockDim.x / WARP_SIZE)) + warp_tile.meta_group_rank();
  // The index of key for this thread
  const int key_idx = (warp_tile_global_idx * task_per_warp_tile) + lane_idx;
  // The assigned key for this lane(thread)
  int key;
  // The dst slabset and the dst slab inside this set
  int src_set;
  int src_slab;
  // The variable that contains the missing key
  int missing_key;
  // The variable that contains the index for the missing key
  uint64_t missing_index;
  // The counter for counting the missing key in this warp
  uint8_t warp_missing_counter = 0;
  // Active flag: whether current lane(thread) has unfinished task
  bool active = false;
  if (lane_idx < task_per_warp_tile) {
    if (key_idx < len) {
      active = true;
      key = d_keys[key_idx];
      src_set = key % capacity_in_set;
      src_slab = key % SET_ASSOCIATIVITY;
    }
  }

  // Lane participate in warp_tile ballot to produce warp-level work queue
  unsigned active_mask = warp_tile.ballot(active);

  // The warp-level outer loop: finish all the tasks within the work queue
  while (active_mask != 0) {
    // Next task in the work quere, start from lower index lane(thread)
    int next_lane = __ffs(active_mask) - 1;
    // Broadcast the task and the global index to all lane in the warp_tile
    int next_key = warp_tile.shfl(key, next_lane);
    int next_idx = warp_tile.shfl(key_idx, next_lane);
    int next_set = warp_tile.shfl(src_set, next_lane);
    int next_slab = warp_tile.shfl(src_slab, next_lane);

    // Counter to record how many slab have been searched
    int counter = 0;

    // Working queue before task started
    const unsigned old_active_mask = active_mask;

    // Lock the slabset before operating the slabset
    //warp_lock_mutex(warp_tile, set_mutex[next_set]);

    // The warp-level inner loop: finish a single task in the work queue
    while (active_mask == old_active_mask) {
      // When all the slabs inside a slabset have been searched, mark missing task, task is
      // completed
      if (counter >= SET_ASSOCIATIVITY) {
        if (lane_idx == warp_missing_counter) {
          missing_key = next_key;
          missing_index = next_idx;
        }

        if (lane_idx == (int)next_lane) {
          active = false;
        }

        warp_missing_counter++;
        active_mask = warp_tile.ballot(active);
        break;
      }

      // The warp_tile read out the slab
      int read_key = ((int*)(keys[next_set].set_[next_slab].slab_))[lane_idx];

      // Compare the slab data with the target key
      int found_lane = __ffs(warp_tile.ballot(read_key == next_key)) - 1;

      // If found, mark hit task, copy the founded data, the task is completed
      if (found_lane >= 0) {
        int found_offset = (next_set * SET_ASSOCIATIVITY + next_slab) * WARP_SIZE + found_lane;
        if (lane_idx == (int)next_lane) {
          slot_counter[found_offset] = atomicAdd(global_counter, 0);
          active = false;
        }

        warp_tile_copy(lane_idx, embedding_vec_size,
                                  (float*)(d_values + next_idx * embedding_vec_size),
                                  (float*)(vals + found_offset * embedding_vec_size));

        active_mask = warp_tile.ballot(active);
        break;
      }

      // Compare the slab data with empty key, if found empty key, mark missing task, task is
      // completed
      if (warp_tile.ballot(read_key == empty_key) != 0) {
        if (lane_idx == warp_missing_counter) {
          missing_key = next_key;
          missing_index = next_idx;
        }

        if (lane_idx == (int)next_lane) {
          active = false;
        }

        warp_missing_counter++;
        active_mask = warp_tile.ballot(active);
        break;
      }

      // Not found in this slab, the task is not completed, goto searching next slab
      counter++;
      next_slab = (next_slab + 1) % SET_ASSOCIATIVITY;
    }

    // Unlock the slabset after operating the slabset
    //warp_unlock_mutex(warp_tile, set_mutex[next_set]);
  }

  // After warp_tile complete the working queue, save the result for output
  // First thread of the warp_tile accumulate the missing length to global variable
  int warp_position;
  if (lane_idx == 0) {
    if(warp_missing_counter > 0){
      warp_position = atomicAdd(d_missing_len, (int)warp_missing_counter);
    }
  }
  warp_position = warp_tile.shfl(warp_position, 0);

  if (lane_idx < warp_missing_counter) {
    d_missing_keys[warp_position + lane_idx] = missing_key;
    d_missing_index[warp_position + lane_idx] = missing_index;
  }

  __threadfence();
  *miss_count = *d_missing_len;
}

__global__ void CopyMissingToOutput(float* output, float* memcpy_buffer_gpu, int *missing_index, int value_len, int miss_count) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int item_id = i / value_len;
  int item_pos = i % value_len;

  if (item_id < miss_count) {
    output[missing_index[item_id]] = memcpy_buffer_gpu[item_id * value_len + item_pos];
  }
}

__global__ void insert_replace_kernel(const int* d_keys, const float* d_values,
                                      const int embedding_vec_size, const int len,
                                      slab_set* keys,  float* vals,
                                      int* slot_counter,
                                      int* set_mutex, int* global_counter,
                                      const int capacity_in_set,
                                      const int task_per_warp_tile) {
  int empty_key = -1;
  // Lane(thread) ID within a warp_tile
  cg::thread_block_tile<WARP_SIZE> warp_tile =
      cg::tiled_partition<WARP_SIZE>(cg::this_thread_block());
  const int lane_idx = warp_tile.thread_rank();
  // Warp tile global ID
  const int warp_tile_global_idx =
      (blockIdx.x * (blockDim.x / WARP_SIZE)) + warp_tile.meta_group_rank();
  // The index of key for this thread
  const int key_idx = (warp_tile_global_idx * task_per_warp_tile) + lane_idx;
  // The assigned key for this lane(thread)
  int key;
  // The dst slabset and the dst slab inside this set
  int src_set;
  int src_slab;
  // Active flag: whether current lane(thread) has unfinished task
  bool active = false;
  if (lane_idx < task_per_warp_tile) {
    if (key_idx < len) {
      active = true;
      key = d_keys[key_idx];
      src_set = key % capacity_in_set;
      src_slab = key % SET_ASSOCIATIVITY;
    }
  }

  // Lane participate in warp_tile ballot to produce warp-level work queue
  unsigned active_mask = warp_tile.ballot(active);

  // The warp-level outer loop: finish all the tasks within the work queue
  while (active_mask != 0) {
    // Next task in the work quere, start from lower index lane(thread)
    int next_lane = __ffs(active_mask) - 1;
    // Broadcast the task, the global index and the src slabset and slab to all lane in a warp_tile
    int next_key = warp_tile.shfl(key, next_lane);
    int next_idx = warp_tile.shfl(key_idx, next_lane);
    int next_set = warp_tile.shfl(src_set, next_lane);
    int next_slab = warp_tile.shfl(src_slab, next_lane);
    int first_slab = next_slab;

    // Counter to record how many slab have been searched
    int counter = 0;

    // Variable to keep the min slot counter during the probing
    int max_int = 9999999;
    int max_slab_distance = 9999999;
    int min_slot_counter_val = max_int;
    // Variable to keep the slab distance for slot with min counter
    int slab_distance = max_slab_distance;
    // Variable to keep the slot distance for slot with min counter within the slab
    int slot_distance;
    // Working queue before task started
    const unsigned old_active_mask = active_mask;

    // Lock the slabset before operating the slabset
    warp_lock_mutex(warp_tile, set_mutex[next_set]);

    // The warp-level inner loop: finish a single task in the work queue
    while (active_mask == old_active_mask) {
      // When all the slabs inside a slabset have been searched
      // and no empty slots or target slots are found. Replace with LRU
      if (counter >= SET_ASSOCIATIVITY) {
        // (sub)Warp all-reduction, the reduction result store in all threads
        warp_min_reduction(warp_tile, min_slot_counter_val,
                                                        slab_distance, slot_distance);

        // Calculate the position of LR slot
        int target_slab = (first_slab + slab_distance) % SET_ASSOCIATIVITY;
        int slot_index =
            (next_set * SET_ASSOCIATIVITY + target_slab) * WARP_SIZE + slot_distance;

        // Replace the LR slot
        if (lane_idx == (int)next_lane) {
          (( int*)(keys[next_set].set_[target_slab].slab_))[slot_distance] = key;
          slot_counter[slot_index] = atomicAdd(global_counter, 0);
        }

        warp_tile_copy(lane_idx, embedding_vec_size,
                                  ( float*)(vals + slot_index * embedding_vec_size),
                                  ( float*)(d_values + next_idx * embedding_vec_size));

        // Replace complete, mark this task completed
        if (lane_idx == (int)next_lane) {
          active = false;
        }

        active_mask = warp_tile.ballot(active);
        break;
      }

      // The warp_tile read out the slab
      int read_key = (( int*)(keys[next_set].set_[next_slab].slab_))[lane_idx];

      // Compare the slab data with the target key
      int found_lane = __ffs(warp_tile.ballot(read_key == next_key)) - 1;

      // If found target key, the insertion/replace is no longer needed.
      // Refresh the slot, the task is completed
      if (found_lane >= 0) {
        int found_offset = (next_set * SET_ASSOCIATIVITY + next_slab) * WARP_SIZE + found_lane;
        if (lane_idx == (int)next_lane) {
          slot_counter[found_offset] = atomicAdd(global_counter, 0);
          active = false;
        }

        active_mask = warp_tile.ballot(active);
        break;
      }

      // Compare the slab data with empty key.
      // If found empty key, do insertion,the task is complete
      found_lane = __ffs(warp_tile.ballot(read_key == empty_key)) - 1;
      if (found_lane >= 0) {
        int found_offset = (next_set * SET_ASSOCIATIVITY + next_slab) * WARP_SIZE + found_lane;

        if (lane_idx == (int)next_lane) {
          (( int*)(keys[next_set].set_[next_slab].slab_))[found_lane] = key;
          slot_counter[found_offset] = atomicAdd(global_counter, 0);
        }

        warp_tile_copy(lane_idx, embedding_vec_size,
                                  ( float*)(vals + found_offset * embedding_vec_size),
                                  ( float*)(d_values + next_idx * embedding_vec_size));

        if (lane_idx == (int)next_lane) {
          active = false;
        }

        active_mask = warp_tile.ballot(active);
        break;
      }

      // If no target or unused slot found in this slab,
      // Refresh LR info, continue probing
      int read_slot_counter =
          slot_counter[(next_set * SET_ASSOCIATIVITY + next_slab) * WARP_SIZE + lane_idx];
      if (read_slot_counter < min_slot_counter_val) {
        min_slot_counter_val = read_slot_counter;
        slab_distance = counter;
      }

      counter++;
      next_slab = (next_slab + 1) % SET_ASSOCIATIVITY;
    }

    // Unlock the slabset after operating the slabset
    warp_unlock_mutex(warp_tile, set_mutex[next_set]);
  }
}

void CEmbeddingMap::InitEmbedding(std::string strFileloc, int bFirstLineDelete){
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
        EmbeddingOnDRAM.emplace_back(tmp);
        vKey.emplace_back(nKeyTmp);
    }

    totalMissCount = 0;
    totalHitCount = 0;
    totalBatch = 0;
    missingBatch = 0;

    //Initialize CPU embedding map
    auto iter2 = EmbeddingOnDRAM.begin();
    for (auto iter1 = vKey.begin(); iter1 != vKey.end(); iter1++) {
        Set(*iter1,&(*iter2));
        iter2++;
    }

    //Calculate parameters for SlabSet GPU Cache
    num_slot_ = CACHE_SIZE;
    capacity_in_set_ = num_slot_ / (SET_ASSOCIATIVITY * WARP_SIZE);
    embedding_vec_size_ = EMBEDDING_DIM;

    // Allocate GPU memory for cache
    cudaMalloc((void **)&keys_, sizeof(slab_set) * capacity_in_set_);
    cudaMalloc((void **)&vals_, sizeof(float) * embedding_vec_size_ * num_slot_);
    cudaMalloc((void **)&slot_counter_, sizeof(int) * num_slot_);
    cudaMalloc((void **)&global_counter_, sizeof(int));

    cudaMalloc((void **)&set_mutex_, sizeof(int) * capacity_in_set_);

    // Initialize GPU embedding map
    // Initialize the cache, set all entry to unused <K,V>
    const int empty_key = -1;
    init_cache<<<((num_slot_ - 1) / BLOCK_SIZE) + 1, BLOCK_SIZE>>>(keys_, slot_counter_, global_counter_, num_slot_, empty_key, set_mutex_, capacity_in_set_);

    ifDataSet.close();
}


void CEmbeddingMap::GatherBatch(const std::vector<int>& line, int cursor, Parameters *gatherResult, int currentBatchSize){ 
    //将Batch中的key拷贝到GPU
    int *keyBatch;
    cudaMalloc((void **)&keyBatch, currentBatchSize * sizeof(int));
    cudaMemcpy(keyBatch, &line[cursor], currentBatchSize * sizeof(int), cudaMemcpyHostToDevice);

    int *d_missing_len;
    cudaMalloc((void **)&d_missing_len, sizeof(int));

    clock_gettime(CLOCK_MONOTONIC, &tStart);
    update_kernel_overflow_ignore<<<1, 1>>>(global_counter_, d_missing_len);

    float* output;
    int *d_missing_index, *d_missing_keys;
    int *miss_count;
    cudaMalloc((void **)&output, currentBatchSize * sizeof(float) * embedding_vec_size_);
    cudaMalloc((void **)&d_missing_index, currentBatchSize * sizeof(int));

    cudaHostAlloc((void **)&d_missing_keys, currentBatchSize * sizeof(int), cudaHostAllocDefault);
    cudaHostAlloc((void **)&miss_count, sizeof(int), cudaHostAllocDefault);

    // Read from the cache
    // Touch and refresh the hitting slot
    const int keys_per_block = (BLOCK_SIZE / WARP_SIZE) * 1;
    const int grid_size = ((currentBatchSize - 1) / keys_per_block) + 1;

    get_kernel<<<grid_size, BLOCK_SIZE>>>(keyBatch, currentBatchSize, output, embedding_vec_size_, d_missing_index, d_missing_keys, d_missing_len, 
            miss_count, global_counter_, slot_counter_, capacity_in_set_, keys_, vals_, set_mutex_, 1);
    cudaDeviceSynchronize();

    clock_gettime(CLOCK_MONOTONIC, &tEnd);
    hitTime += ((double)(tEnd.tv_sec - tStart.tv_sec)*1000000000 + tEnd.tv_nsec - tStart.tv_nsec)/1000000;
    
    if(*miss_count > 0){
        missingBatch++;
        float *memcpy_buffer_gpu;
        cudaHostAlloc((void **)&memcpy_buffer_gpu, currentBatchSize * embedding_vec_size_ * sizeof(float), cudaHostAllocWriteCombined);

        //从CPU中查找缺失的Embedding
        //TODO::修改为多线程查找
        clock_gettime(CLOCK_MONOTONIC, &tStart);
        for(int i = 0; i < *miss_count; i++){
            Parameters *tmp;
            tmp = Get(d_missing_keys[i]);
            for(int j = 0;j < EMBEDDING_DIM; j++){
                memcpy_buffer_gpu[i * EMBEDDING_DIM + j] = tmp->a[j];
            }
        }
        clock_gettime(CLOCK_MONOTONIC, &tEnd);
        lookUpTime += ((double)(tEnd.tv_sec - tStart.tv_sec)*1000000000 + tEnd.tv_nsec - tStart.tv_nsec)/1000000;

        //将查询结果拷上GPU
        clock_gettime(CLOCK_MONOTONIC, &tStart);
        CopyMissingToOutput<<<(*miss_count * embedding_vec_size_ - 1) / BLOCK_SIZE + 1, BLOCK_SIZE>>>(output, memcpy_buffer_gpu, d_missing_index, embedding_vec_size_, *miss_count);
        clock_gettime(CLOCK_MONOTONIC, &tEnd);
        memcpyTime += ((double)(tEnd.tv_sec - tStart.tv_sec)*1000000000 + tEnd.tv_nsec - tStart.tv_nsec)/1000000;

        insert_replace_kernel<<<((currentBatchSize - 1) / keys_per_block) + 1, ((*miss_count - 1) / keys_per_block) + 1>>>(d_missing_keys, memcpy_buffer_gpu, 
                    embedding_vec_size_, *miss_count, keys_, vals_, slot_counter_, set_mutex_, global_counter_, capacity_in_set_, 1);
    }   

    totalHitCount += currentBatchSize - *miss_count;
    totalMissCount += *miss_count;
    totalBatch++;

    cudaFree(keyBatch);
    cudaFree(d_missing_len);
    cudaFree(d_missing_index);
    cudaFreeHost(d_missing_keys);
    cudaFreeHost(miss_count);
}

void CEmbeddingMap::GatherWork(const std::vector<int>& line, Parameters *gatherResult){
    int cursor = 0;
    int end = line.size();
    hitTime = 0;
    statusMemcpyTime = 0;
    lookUpTime = 0;
    memcpyTime = 0;

    while(end - cursor >= BATCH_SIZE){
        GatherBatch(line, cursor, gatherResult, BATCH_SIZE);
        cursor += BATCH_SIZE;
    }
    GatherBatch(line, cursor, gatherResult, end - cursor);
}

float CEmbeddingMap::GetHitRate(){
    return totalHitCount / (totalHitCount + totalMissCount);
}

float CEmbeddingMap::GetMissingBatchRate(){
    return missingBatch / totalBatch;
}

float CEmbeddingMap::GetHitTime(){
    return hitTime;
}

float CEmbeddingMap::GetStatusMemcpyTime(){
    return statusMemcpyTime;
}

float CEmbeddingMap::GetLookUpTime(){
    return lookUpTime;
}

float CEmbeddingMap::GetMemcpyTime(){
    return memcpyTime;
}

void CEmbeddingMap::MoveAllEmbeddings(Parameters *CPUEmbeddingAddress){
   
}

void CEmbeddingMap::DeleteEmbedding(){
    cudaFree(keys_);
    cudaFree(vals_);
    cudaFree(set_mutex_);
    cudaFree(slot_counter_);
    cudaFree(global_counter_);
}

