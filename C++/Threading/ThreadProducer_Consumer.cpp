#include <iostream>
#include <thread>
#include <queue>
#include <condition_variable>
#include <chrono>

std::queue<int> buffer;
const unsigned int maxBufferSize = 10;
std::mutex mtx;
std::condition_variable bufferNotEmpty;
std::condition_variable bufferNotFull;

void producer() {
    for (int i = 0; i < 20; ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100)); // Simulate work
        std::unique_lock<std::mutex> lock(mtx);
        bufferNotFull.wait(lock, [] { return buffer.size() < maxBufferSize; });
        buffer.push(i);
        std::cout << "Produced: " << i << std::endl;
        bufferNotEmpty.notify_one();
    }
}

void consumer() {
    for (int i = 0; i < 20; ++i) {
        std::unique_lock<std::mutex> lock(mtx);
        bufferNotEmpty.wait(lock, [] { return !buffer.empty(); });
        int value = buffer.front();
        buffer.pop();
        std::cout << "Consumed: " << value << std::endl;
        bufferNotFull.notify_one();
    }
}

int main() {
    std::thread prod(producer);
    std::thread cons(consumer);

    prod.join();
    cons.join();

    return 0;
}
