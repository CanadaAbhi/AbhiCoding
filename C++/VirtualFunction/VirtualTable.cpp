#include <iostream>

class Animal {
    public:
        virtual void sound() {
            std::cout << "Animal sound\n";
        }
    };

class Dog : public Animal {
    public:
        void sound() override {  // Overrides base class's virtual function
            std::cout << "Dog barks\n";
        }
    };

int main() {
        Animal* a = new Dog();  // Base pointer to derived object
        a->sound();             // Calls Dog::sound() at runtime (late binding)
        delete a;
}

class Cat : public Animal {
    public:
        void sound() override {
            std::cout << "Cat meows\n";
        }
    };
    
int main() {
        Animal* a1 = new Dog();
        Animal* a2 = new Cat();
        a1->sound();  // Dog barks
        a2->sound();  // Cat meows
        delete a1;
        delete a2;
}
    