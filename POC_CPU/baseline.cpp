#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream> 
#include <unordered_map>
#include "embedding_map/embedding_map.h"
#include "../time/timecalculate.h"
#include "../FileRW/csvRW.h"

#define DataSize 26557961

int main() {
	std::ofstream ofEmbeddingMap;
	CTimeCalculate iTimeCal;

	std::vector<int> line;

/* 	//将所需数据存到lines向量中
	iTimeCal.StartWork("reading");
	std::string dataloc =  "../dataset/ad_feature.csv";
	readcsv(dataloc,line,1,1);
	iTimeCal.EndWork("reading"); */

	//换取另一个数据集
	std::ifstream ifs;
	iTimeCal.StartWork("reading");
	ifs.open("../dataset/adgroup_id.txt");
	int temp;
	for(int i = 0; i < DataSize; i++){
		ifs >> temp;
		line.push_back(temp);
	}
	ifs.close();
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

	//将更新后的embedding写入csv文件中验证
	iTimeCal.StartWork("storing");
	ofEmbeddingMap.open("embedding_map/ofembedding.csv");
	for (auto iter = em.a_map.begin(); iter != em.a_map.cend(); iter++) {
		ofEmbeddingMap << (*iter). first << "," << (*iter).second->a[1]<< "," << (*iter).second->v[1]<< "\n";
	} 
	ofEmbeddingMap.close();
	iTimeCal.EndWork("storing");
	*/
	
	iTimeCal.StartWork("Gathering");
	int totalLength = line.size();
	Parameters *gatherResult = new Parameters[totalLength];
	em.MultiThreadGatherEV(line, gatherResult);
	iTimeCal.EndWork("Gathering");

	iTimeCal.StartWork("storing");
	ofEmbeddingMap.open("embedding_map/ofembedding.csv");
	ofEmbeddingMap << "key,a,v\n";
	for (int i = 0; i < totalLength; i++) {
		ofEmbeddingMap << line[i] << "," << gatherResult[i].a[1] << "," << gatherResult[i].v[1] << "\n";
	} 
	ofEmbeddingMap.close();
	iTimeCal.EndWork("storing");

	delete []gatherResult;
	
	return 0;
}
