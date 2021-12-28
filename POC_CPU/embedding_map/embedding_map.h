#pragma once
#include <iostream>
#include <unordered_map>
#include <fstream>
#include <vector>
#include <mutex>
#include <thread>
#include <math.h>
#include <time.h>

struct parameters{
	float a;	
	float v; 	//embedding
};

struct time_interval{
	timespec memstart, memend;
	float memcpy_time_1,memcpy_time_2,update_time;
};//ti用于记录每个线程中的各项任务的时间

class embedding_map{
private:
	std::mutex a_mutex;
	const int g = 1;
	const int c = 1;
public:
	std::unordered_map<int, parameters *> a_map;
	parameters* get(int Key) {
		std::lock_guard<std::mutex> guard(a_mutex);

		return a_map.at(Key);
	};

	void set(int Key, parameters* Value) {
		std::lock_guard<std::mutex> guard(a_mutex);
		a_map.insert(std::make_pair(Key, Value)); 
	};

	void erase(int key)
	{
		std::lock_guard<std::mutex> guard(a_mutex);
		a_map.erase(key);
	}

	void initembedding(std::string fileloc,std::vector<parameters> &lines,int firstlinedelete){
		std::ifstream dataset;
		dataset.open(fileloc);
	
		std::string lineStr;
		int column_data;
		char comma;
		int keytmp;
		std::vector<int> key;
	 
		if(firstlinedelete){
			std::getline(dataset, lineStr);
		}

		while (std::getline(dataset, lineStr))
		{
			std::stringstream ss(lineStr);
			std::string str;
			parameters tmp;
			ss >> keytmp;
			ss >> comma;
			ss >> tmp.a;
			ss >> comma;
			ss >> tmp.v;
			lines.emplace_back(tmp);
			key.emplace_back(keytmp);
			//std::cout << key << "," << iter.a <<","<< iter.v << ","<< &iter << std::endl;
		}
		auto iter2 = lines.begin();
		for (auto iter1 = key.begin(); iter1 != key.end(); iter1++) {
			set(*iter1,&(*iter2));
			iter2++;
		}

		dataset.close();
	}

	void batch_work(const std::vector<int>& line,int cursor,parameters *batch,int current_batch_size,time_interval &ti){
		parameters* tmp;
		int batchcursor = 0;

		//memcpy,将查询到的数据复制到连续的batch空间中
		clock_gettime(CLOCK_MONOTONIC, &ti.memstart);
		for (auto iter = line.cbegin() + cursor; iter != line.cbegin() + cursor + current_batch_size; iter++) {
			tmp = get(*iter);
			batch[batchcursor].a = tmp->a;
			batch[batchcursor].v = tmp->v;
			batchcursor++;
		}
		clock_gettime(CLOCK_MONOTONIC, &ti.memend);
		ti.memcpy_time_1 += ((double)(ti.memend.tv_sec - ti.memstart.tv_sec)*1000000000 + ti.memend.tv_nsec - ti.memstart.tv_nsec)/1000000;


		//计算更新embedding
		clock_gettime(CLOCK_MONOTONIC, &ti.memstart);
		for(int i = 0;i < current_batch_size;++i){
			batch[i].a += g * g;
			batch[i].v -= (c * g * 1.0) / sqrt(batch[i].a);
		}
		clock_gettime(CLOCK_MONOTONIC, &ti.memend);
		ti.update_time += ((double)(ti.memend.tv_sec - ti.memstart.tv_sec)*1000000000 + ti.memend.tv_nsec - ti.memstart.tv_nsec)/1000000;

		//memcpy，将更新后的数据拷回
		batchcursor = 0;
		clock_gettime(CLOCK_MONOTONIC, &ti.memstart);
		for(auto iter = line.cbegin() + cursor; iter != line.cbegin() + cursor + current_batch_size; iter++) {
			tmp = get(*iter);
			tmp->a = batch[batchcursor].a ;
			tmp->v = batch[batchcursor].v ;
			batchcursor++;
		}
		clock_gettime(CLOCK_MONOTONIC, &ti.memend);
		ti.memcpy_time_2 += ((double)(ti.memend.tv_sec - ti.memstart.tv_sec)*1000000000 + ti.memend.tv_nsec - ti.memstart.tv_nsec)/1000000;
	}


	void work(const std::vector<int>& line,int start,int end,int worker_id)
	{	
		parameters* tmp;
		int cursor = start;
		int batchcursor = 0;
		const int batchSize = 256 * 64;
		parameters *batch= new parameters[batchSize];
		time_interval ti;
		ti.memcpy_time_1 = 0;
		ti.memcpy_time_2 = 0;
		ti.update_time = 0;

		while(end - cursor >= batchSize){
			batch_work(line,cursor,batch,batchSize,ti);
			cursor += batchSize;
		}

		batch_work(line,cursor,batch,end - cursor,ti);

		delete []batch;

		std::cout << "线程" << worker_id << "已经结束" << std::endl;
		std::cout << "memcpy time 1:" << ti.memcpy_time_1 << "ms" << std::endl;
		std::cout << "memcpy time 2:" << ti.memcpy_time_2 << "ms" << std::endl;
		std::cout << "update time:" << ti.update_time << "ms" << std::endl;
	}

	void updateembedding(const std::vector<int>& line) {

		auto n = std::thread::hardware_concurrency();
		std::cout << "CPU核心数为:\t" << n << std::endl;

		const int nThreadNum = 4; 
		int nScope = line.size() / nThreadNum;

		std::thread th_arr[nThreadNum];

		for (unsigned int i = 0; i < nThreadNum - 1; ++i) {
			th_arr[i] = std::thread(&embedding_map::work,this, std::ref(line), i * nScope, (i + 1) * nScope, i);
		}
		th_arr[nThreadNum - 1] = std::thread(&embedding_map::work,this, std::ref(line), (nThreadNum - 1) * nScope, line.size(), nThreadNum - 1);
		for (unsigned int i = 0; i < nThreadNum; ++i) {
			th_arr[i].join();
		}
	}
};