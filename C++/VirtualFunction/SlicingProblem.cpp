#include <iostream>
class BaseAnimal {
    public:
        virtual void info() {
            std::cout << "BaseAnimal\n";
        }
    };
    
    class DogAnimal : public BaseAnimal {
    public:
        void info() override {
            std::cout << "DogAnimal\n";
        }
    
        void onlyInDog() {
            std::cout << "Unique to Dog\n";
        }
    };
    
int main() {
        DogAnimal dog;
        BaseAnimal base = dog;  //  Object slicing — Dog parts lost
        base.info();            // Outputs: BaseAnimal
}
    