#include <iostream>

class DynamicVector {
public:
    DynamicVector() : capacity(2), size(0) {
        array = new int[capacity];
    }

    ~DynamicVector() {
        delete[] array;
    }

    void insert(int value) {
        if (size == capacity) {
            resize();
        }
        array[size++] = value;
    }

    void display() const {
        for (int i = 0; i < size; ++i) {
            std::cout << array[i] << " ";
        }
        std::cout << std::endl;
    }

private:
    void resize() {
        capacity *= 2;
        int* new_array = new int[capacity];
        for (int i = 0; i < size; ++i) {
            new_array[i] = array[i];
        }
        delete[] array;
        array = new_array;
    }

    int* array;
    int capacity;
    int size;
};

int main() {
    DynamicVector vec;
    vec.insert(1);
    vec.insert(2);
    vec.insert(3);
    vec.insert(4);
    vec.insert(5);

    std::cout << "Dynamic vector elements: ";
    vec.display();

    return 0;
}
