#pragma once
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream> 

struct parameters{
	int key;
	float a;	
	float v; 	//embedding
};

//从fileloc位置读取embedding文件存储到容器中，firstline表示是否删除第一行信息 
void readembedding(std::string fileloc,std::vector<parameters> &lines,int firstlinedelete){
	std::ifstream dataset;
	dataset.open(fileloc);
	
	std::string lineStr;
	int column_data;
	char comma;
	 
	if(firstlinedelete){
		std::getline(dataset, lineStr);
	}
	
	while (std::getline(dataset, lineStr))
	{
		std::stringstream ss(lineStr);
		std::string str;
		parameters tmp;
		ss >> tmp.key;
		ss >> comma;
		ss >> tmp.a;
		ss >> comma;
		ss >> tmp.v;
		lines.emplace_back(tmp);
	}
	dataset.close();
}

void writeembedding(std::string fileloc,std::vector<parameters> &lines,int firstlineadd){
	std::ofstream embedding;
	embedding.open(fileloc);
	
	std::string lineStr;

	 
	if(firstlineadd){
		embedding << "key,a,v" << std::endl;
	}


	for (auto iter = lines.cbegin(); iter != lines.cend(); iter++) {
		embedding << iter->key << ',' << iter->a << ',' << iter->v << std::endl;
	}
	embedding.close();
}