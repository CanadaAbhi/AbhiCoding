#include <iostream>
class Shape {
    public:
        virtual void draw() = 0;  // Pure virtual
    };

class Circle : public Shape {
    public:
        void draw() override {
            std::cout << "Drawing Circle\n";
        }
    };
    
    int main() {
        // Shape s;  Error: Cannot instantiate abstract class
        Shape* shape = new Circle();
        shape->draw();
        delete shape;
}
    