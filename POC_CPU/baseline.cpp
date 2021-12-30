#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream> 
#include <unordered_map>
#include "embedding_map/embedding_map.h"
#include "time/timecalculate.h"
#include "FileRW/csvRW.h"

int main() {
	std::ofstream ofEmbeddingMap;
	CTimeCalculate iTimeCal;

	std::vector<int> vLines;

	//将所需数据存到vLines向量中
	iTimeCal.StartWork("reading");
	
	std::string dataloc =  "../dataset/ad_feature.csv";
	readcsv(dataloc,vLines,1,1);

	iTimeCal.EndWork("reading");

	//从文件读取embedding向量，并建立hashmap
	iTimeCal.StartWork("initialzing");

	CEmbeddingMap em;
	std::vector<Parameters> em_paras;
	em.InitEmbedding("embedding_map/embedding.csv",em_paras,1);

	iTimeCal.EndWork("initialzing");
	
	//更新embedding表
	iTimeCal.StartWork("updating");

	em.UpdateEV(vLines);

	iTimeCal.EndWork("updating");

	//将更新后的embedding写入csv文件中验证
	iTimeCal.StartWork("storing");

	ofEmbeddingMap.open("embedding_map/ofembedding.csv");
	for (auto iter = em.a_map.begin(); iter != em.a_map.cend(); iter++) {
		ofEmbeddingMap << (*iter). first << "," << (*iter).second->a[1]<< "," << (*iter).second->v[1]<< "\n";
	} 
	ofEmbeddingMap.close();

	iTimeCal.EndWork("storing");

	return 0;
}
