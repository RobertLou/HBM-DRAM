#pragma once
#include <iostream>
#include <unordered_map>
#include <fstream>
#include <vector>
#include <mutex>
#include <thread>
#include <math.h>

struct parameters{
	float a;	
	float v; 	//embedding
};



class embedding_map{
private:
	std::mutex a_mutex;
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



	void work(const std::vector<int>& line,int start,int end,int worker_id)
	{	
		parameters* tmp;
		int g = 1;
		int c = 1;
		for (auto iter = line.cbegin() + start; iter != line.cbegin() + end; iter++) {
			tmp = get(*iter);
			
			tmp->a+=g*g;
			tmp->v-=(c*g*1.0)/sqrt(tmp->a);
			/*
			for (int i = 0; i < 10000; i++) {
				tmp->a+=1;
				tmp->v+=1;
			}
			*/
			
		}
		std::cout << "线程" << worker_id << "已经结束" << std::endl;
	}

	void updateembedding(const std::vector<int>& line) {

		auto n = std::thread::hardware_concurrency();
		std::cout << "CPU核心数为:\t" << n << std::endl;

		const int nThreadNum = 16; 
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