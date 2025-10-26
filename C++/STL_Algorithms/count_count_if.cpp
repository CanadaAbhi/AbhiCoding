#include <iostream>
#include <vector>
#include <algorithm>

int main() {
    std::vector<int> v = {1, 2, 2, 3, 4};
    int count_twos = std::count(v.begin(), v.end(), 2);
    int count_even = std::count_if(v.begin(), v.end(), [](int x){ return x % 2 == 0; });
    std::cout << "Twos: " << count_twos << ", Even: " << count_even << std::endl;
}
