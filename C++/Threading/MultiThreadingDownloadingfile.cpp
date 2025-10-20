#include <iostream>
#include <thread>

void downloadFile(const std::string& filename) {
    std::cout << "Downloading: " << filename << "\n";
    std::this_thread::sleep_for(std::chrono::seconds(2));
    std::cout << "Finished: " << filename << "\n";
}

int main() {
    std::thread t1(downloadFile, "file1.txt");
    std::thread t2(downloadFile, "file2.txt");
    t1.join();
    t2.join();
}
