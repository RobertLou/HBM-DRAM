#pragma once

#include <iostream>
#include <ctime>
#include <time.h>
#include <string>

class CTimeCalculate {
public:
	timespec tStart, tEnd;

	void StartWork(std::string str) {
		std::cout << "start " << str <<"......" << std::endl;
		clock_gettime(CLOCK_MONOTONIC, &tStart);
	}
	void EndWork(std::string str) {
		clock_gettime(CLOCK_MONOTONIC, &tEnd);
		std::cout << str << " time: " << ((double)(tEnd.tv_sec - tStart.tv_sec)*1000000000 + tEnd.tv_nsec - tStart.tv_nsec)/1000000 << "ms" << std::endl;
		std::cout << "end " << str << "......\n\n"  << std::endl;
	}
};