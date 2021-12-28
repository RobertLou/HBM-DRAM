#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream> 
#include <unordered_map>
#include "embedding_map/embedding_map.h"
#include "time/timecalculate.h"
#include "FileRW/csvRW.h"

template <class T> 
void showvec(const std::vector<T>& line) {
	for (auto iter = line.cbegin(); iter != line.cend(); iter++) {
		std::cout << (*iter) << std::endl;
	}
};

int main() {
	std::ofstream ofembedding;
	std::string lineStr;
	timecalculate timecal;

	std::vector<int> lines;
	
	//将所需数据存到lines向量中
	timecal.startwork("reading");
	
	std::string dataloc =  "../dataset/ad_feature.csv";
	readcsv(dataloc,lines,1,1);

	timecal.endwork("reading");

	//从文件读取embedding向量，并建立hashmap
	timecal.startwork("initialzing");

	embedding_map em;
	std::vector<parameters> em_paras;
	em.initembedding("embedding_map/embedding.csv",em_paras,1);

	timecal.endwork("initialzing");
	
	//更新embedding表
	timecal.startwork("updating");

	em.updateembedding(lines);

	timecal.endwork("updating");

	//将更新后的embedding写入csv文件中验证
	timecal.startwork("storing");

	ofembedding.open("embedding_map/ofembedding.csv");
	for (auto iter = em.a_map.begin(); iter != em.a_map.cend(); iter++) {
		ofembedding << (*iter). first << "," << (*iter).second->a<< "," << (*iter).second->v<< "\n";
	} 
	ofembedding.close();

	timecal.endwork("storing");


	return 0;
}
