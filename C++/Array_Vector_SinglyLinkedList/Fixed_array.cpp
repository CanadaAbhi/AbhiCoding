#include <iostream>

class FixedArray {
public:
    FixedArray(int size) : size(size), index(0) {
        array = new int[size];
    }

    ~FixedArray() {
        delete[] array;
    }

    void insert(int value) {
        if (index < size) {
            array[index++] = value;
        } else {
            std::cout << "Array is full!" << std::endl;
        }
    }

    void display() const {
        for (int i = 0; i < index; ++i) {
            std::cout << array[i] << " ";
        }
        std::cout << std::endl;
    }

private:
    int* array;
    int size;
    int index;
};

int main() {
    FixedArray arr(5);
    arr.insert(1);
    arr.insert(2);
    arr.insert(3);
    arr.insert(4);
    arr.insert(5);
    arr.insert(6); // Should display "Array is full!"

    std::cout << "Fixed-size array elements: ";
    arr.display();

    return 0;
}
