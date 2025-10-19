#include <iostream>
#include <vector>
#include <algorithm>

class IsEven {
public:
    bool operator()(int num) const {
        return num % 2 == 0;
    }
};

int main() {
    std::vector<int> nums = {1,2,3,4,5,6};
    int count = std::count_if(nums.begin(), nums.end(), IsEven());
    std::cout << "Number of even elements: " << count << std::endl; // Output: 3
}
