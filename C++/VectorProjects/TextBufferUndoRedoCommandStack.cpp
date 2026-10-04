// vec_undo.cpp (g++ -O2 -std=c++17 vec_undo.cpp)
#include <vector>
#include <string>
#include <memory>
#include <cstdio>

struct Buffer { std::vector<char> text; };

struct Command {
    virtual ~Command() = default;
    virtual void apply(Buffer&) = 0;
    virtual void revert(Buffer&) = 0;
};

struct Insert : Command {
    size_t pos; std::string s;
    Insert(size_t p, std::string str) : pos(p), s(std::move(str)) {}
    void apply(Buffer& b) override {
        b.text.insert(b.text.begin() + pos, s.begin(), s.end());
    }
    void revert(Buffer& b) override {
        b.text.erase(b.text.begin() + pos, b.text.begin() + pos + s.size());
    }
};

struct Erase : Command {
    size_t pos, len; std::string saved;
    Erase(size_t p, size_t l) : pos(p), len(l) {}
    void apply(Buffer& b) override {
        saved.assign(b.text.begin() + pos, b.text.begin() + pos + len);
        b.text.erase(b.text.begin() + pos, b.text.begin() + pos + len);
    }
    void revert(Buffer& b) override {
        b.text.insert(b.text.begin() + pos, saved.begin(), saved.end());
    }
};

class Editor {
    Buffer buf_;
    std::vector<std::unique_ptr<Command>> undo_, redo_;
public:
    void exec(std::unique_ptr<Command> c) {
        c->apply(buf_);
        undo_.push_back(std::move(c));
        redo_.clear();                        // new edit invalidates redo
    }
    bool undo() {
        if (undo_.empty()) return false;
        undo_.back()->revert(buf_);
        redo_.push_back(std::move(undo_.back()));
        undo_.pop_back();
        return true;
    }
    bool redo() {
        if (redo_.empty()) return false;
        redo_.back()->apply(buf_);
        undo_.push_back(std::move(redo_.back()));
        redo_.pop_back();
        return true;
    }
    void print() const {
        std::printf("[%.*s]\n", (int)buf_.text.size(), buf_.text.data());
    }
};

int main() {
    Editor e;
    e.exec(std::make_unique<Insert>(0, "hello world"));
    e.exec(std::make_unique<Erase>(5, 6));
    e.exec(std::make_unique<Insert>(5, ", kernel"));
    e.print();            // [hello, kernel]
    e.undo(); e.undo();   // back to "hello world"
    e.print();
    e.redo();
    e.print();            // [hello]
}
