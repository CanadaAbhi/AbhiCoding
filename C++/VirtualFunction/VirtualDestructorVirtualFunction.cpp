#include<iostream>
class Base {
    public:
        virtual ~Base() {
            std::cout << "Base destructor\n";
        }
    };
    
    class Derived : public Base {
    public:
        ~Derived() {
            std::cout << "Derived destructor\n";
        }
    };

class Shape {
        public:
            virtual void draw() = 0;  // Pure virtual
};
int main() {
        Base* obj = new Derived();
        delete obj;  // Ensures Derived's destructor is called
}


        