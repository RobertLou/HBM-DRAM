#include "csvRW.h"

int main() {
	std::string fileloc;
	fileloc = "../../../dataset/ad_feature.csv";
	
	std::vector<int> lines;
	
	readcsv(fileloc,lines,3,1);

	std::ofstream a;
	a.open("a.txt"); 
	for (auto iter = lines.cbegin(); iter != lines.cend(); iter++) {
		a << *iter << std::endl;
	}
	a.close();
}
