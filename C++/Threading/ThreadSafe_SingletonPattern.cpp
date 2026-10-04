#include <iostream>
#include <thread>
#include <mutex>

class Singleton {
public:
    static Singleton& getInstance() {
        std::call_once(initInstanceFlag, &Singleton::initInstance);
        return *instance;
    }

    void showMessage() {
        std::cout << "Hello from Singleton!" << std::endl;
    }

private:
    Singleton() {}
    ~Singleton() {}

    static void initInstance() {
        instance = new Singleton;
    }

    static Singleton* instance;
    static std::once_flag initInstanceFlag;
};

Singleton* Singleton::instance = nullptr;
std::once_flag Singleton::initInstanceFlag;

void threadFunc() {
    Singleton::getInstance().showMessage();
}

int main() {
    std::thread t1(threadFunc);
    std::thread t2(threadFunc);

    t1.join();
    t2.join();

    return 0;
}
