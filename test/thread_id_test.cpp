#include <iostream>
#include <thread>
#include <unordered_map>
#include <mutex>

std::unordered_map<std::thread::id, int> threadIDMap;
std::mutex mtx;

int getMappedValueForThread() {
    std::lock_guard<std::mutex> lock(mtx);
    std::thread::id this_id = std::this_thread::get_id();

    if (threadIDMap.find(this_id) == threadIDMap.end()) {
        // 第一次访问该线程，建立映射关系
        static int nextIndex = 0;
        threadIDMap[this_id] = nextIndex++;
    }

    return threadIDMap[this_id];
}

int main() {
    std::cout << "Thread ID mapping: " << std::endl;

    // 创建一些线程并显示它们的映射值
    std::thread t1([]() {
        std::cout << "Thread 1 mapping: " << getMappedValueForThread() << std::endl;
    });

    std::thread t2([]() {
        std::cout << "Thread 2 mapping: " << getMappedValueForThread() << std::endl;
    });

    std::thread t3([]() {
        std::cout << "Thread 3 mapping: " << getMappedValueForThread() << std::endl;
    });

    t1.join();
    t2.join();
    t3.join();

    return 0;
}
