#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream> 
#include <unordered_map>
#include "embedding_map/embedding_map.h"
#include "../time/timecalculate.h"
#include "../FileRW/csvRW.h"

int main() {
	std::ofstream ofEmbeddingMap;
	CTimeCalculate iTimeCal;

	std::vector<int> line;

	//将所需数据存到vLines向量中
	iTimeCal.StartWork("reading");
	std::string dataloc =  "../dataset/ad_feature.csv";
	readcsv(dataloc,line,1,1);
	iTimeCal.EndWork("reading");

	//从文件读取embedding向量，并建立hashmap
	iTimeCal.StartWork("initialzing");
	CEmbeddingMap em;
	std::vector<Parameters> em_paras;
	em.InitEmbedding("embedding_map/embedding.csv",em_paras,1);
	iTimeCal.EndWork("initialzing");
	
	/*
	//更新embedding表
	iTimeCal.StartWork("updating");
	em.MultiThreadUpdateEV(line);
	iTimeCal.EndWork("updating");
	*/

	iTimeCal.StartWork("Gathering");
	int totalLength = line.size();
	Parameters *gatherResult = new Parameters[totalLength];
	em.GatherWork(line, gatherResult);
	iTimeCal.EndWork("Gathering");

	iTimeCal.StartWork("storing");
	ofEmbeddingMap.open("embedding_map/ofembedding.csv");
	ofEmbeddingMap << "key,a,v\n";
	for (int i = 0; i < totalLength; i++) {
		ofEmbeddingMap << gatherResult[i].key << "," << gatherResult[i].a[1] << "," << gatherResult[i].v[1] << "\n";
	} 
	ofEmbeddingMap.close();
	iTimeCal.EndWork("storing");
	delete[] gatherResult;
	
	
	/*
	Parameters *CPUEmbeddingAddress = new Parameters[CACHE_SIZE];
	std::ofstream cacheEmbedding;
	em.MoveAllEmbeddings(CPUEmbeddingAddress);
	cacheEmbedding.open("embedding_map/cachedembedding.csv");
	for (int i = 0; i < CACHE_SIZE; i++) {
		cacheEmbedding << CPUEmbeddingAddress[i].key << "," << CPUEmbeddingAddress[i].a[1] << "," << CPUEmbeddingAddress[i].v[1] << "\n";
	} 
	cacheEmbedding.close();
	delete[] CPUEmbeddingAddress;
	*/

	em.DeleteEmbedding();

	return 0;
}