#include "../FileRW/csvRW.h"

int main()
{
    std::vector<int> lines;
    std::string dataloc =  "../../../dataset/ad_feature.csv";
	readcsv(dataloc,lines,1,1);

    std::ofstream embedding;
	embedding.open("embedding.csv"); 
    embedding << "key,a,v" << std::endl;
	int i = 0;
	for (auto iter = lines.cbegin(); iter != lines.cend(); iter++) {
		embedding << *iter << ',' << i << ',' << i << std::endl;
		i++;
	}
	embedding.close();

}
