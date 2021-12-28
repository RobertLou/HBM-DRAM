#pragma once

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream> 


//从fileloc位置读取csv文件第column列存储到lines容器中，firstline表示是否删除第一行信息 
void readcsv(std::string fileloc,std::vector<int> &lines,int column,int firstlinedelete){
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
		for(int i = 1;i < column;i++){
			ss >> column_data;
			ss >> comma;
		}
		ss >> column_data;
		lines.push_back(column_data);
	}
	
	dataset.close();
}
