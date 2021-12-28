#pragma once

#include <iostream>
#include <ctime>
#include <time.h>
#include <string>

class timecalculate {
public:
	timespec start, end;

	void startwork(std::string str) {
		std::cout << "start " << str <<"......" << std::endl;
		clock_gettime(CLOCK_MONOTONIC, &start);
	}
	void endwork(std::string str) {
		clock_gettime(CLOCK_MONOTONIC, &end);
		std::cout << str << " time: " << ((double)(end.tv_sec - start.tv_sec)*1000000000 + end.tv_nsec - start.tv_nsec)/1000000 << "ms" << std::endl;
		std::cout << "end " << str << "......\n\n"  << std::endl;
	}
};