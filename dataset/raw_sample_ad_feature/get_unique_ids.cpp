#include <iostream>
#include <fstream>
#include <unordered_set>

int main() {
    // 输入文件名和输出文件名
    std::string inputFileName = "restore_keys.txt";
    std::string outputFileName = "output.txt";

    std::ifstream inputFile(inputFileName);
    std::ofstream outputFile(outputFileName);

    if (!inputFile.is_open()) {
        std::cerr << "无法打开输入文件 " << inputFileName << std::endl;
        return 1;
    }

    if (!outputFile.is_open()) {
        std::cerr << "无法打开输出文件 " << outputFileName << std::endl;
        return 1;
    }

    // 用于存储数字的无序集合
    std::unordered_set<int> uniqueNumbers;

    int number;
    while (inputFile >> number) {
        // 将数字添加到无序集合，如果它已经存在，不会重复添加
        uniqueNumbers.insert(number);
    }

    // 将无序集合中的数字写入输出文件
    for (int num : uniqueNumbers) {
        outputFile << num << std::endl;
    }

    inputFile.close();
    outputFile.close();

    std::cout << "已成功删除重复值并写入到 " << outputFileName << std::endl;

    return 0;
}