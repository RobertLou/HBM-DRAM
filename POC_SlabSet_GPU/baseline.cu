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
	em.InitEmbedding("embedding_map/embedding.csv", 1);
	iTimeCal.EndWork("initialzing");
	
	iTimeCal.StartWork("Gathering");
	int totalLength = line.size();
	Parameters *gatherResult = new Parameters[totalLength];
	em.GatherWork(line, gatherResult);
	iTimeCal.EndWork("Gathering");

	std::cout << "Hit Rate:" << em.GetHitRate() << std::endl;
	std::cout << "Missing Batch Rate:" << em.GetMissingBatchRate() << std::endl;
	std::cout << "Hit Time:" << em.GetHitTime() << "ms" << std::endl;
	std::cout << "Status Memcpy Time:" << em.GetStatusMemcpyTime() << "ms" << std::endl;
	std::cout << "LookUp Time:" << em.GetLookUpTime() << "ms" << std::endl;
	std::cout << "Memcpy Time:" << em.GetMemcpyTime() << "ms\n" << std::endl;	

	em.DeleteEmbedding();

	return 0;
}
