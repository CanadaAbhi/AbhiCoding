// vec_iova.cpp (g++ -O2 -std=c++17 vec_iova.cpp)
#include <vector>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <optional>

struct Range { uint64_t start, len; };   // allocated [start, start+len)

class IovaAllocator {
    std::vector<Range> allocs_;          // sorted by start
    uint64_t base_, limit_;
public:
    IovaAllocator(uint64_t base, uint64_t limit) : base_(base), limit_(limit) {}

    std::optional<uint64_t> alloc(uint64_t len) {
        uint64_t candidate = base_;
        auto it = allocs_.begin();
        for (; it != allocs_.end(); ++it) {         // first-fit gap scan
            if (it->start - candidate >= len) break;
            candidate = it->start + it->len;
        }
        if (candidate + len > limit_) return std::nullopt;
        allocs_.insert(it, {candidate, len});       // keeps vector sorted
        return candidate;
    }

    bool free(uint64_t start) {
        auto it = std::lower_bound(allocs_.begin(), allocs_.end(), start,
            [](const Range& r, uint64_t s) { return r.start < s; });
        if (it == allocs_.end() || it->start != start) return false;
        allocs_.erase(it);
        return true;
    }

    bool translate_ok(uint64_t addr) const {        // "TLB lookup"
        auto it = std::upper_bound(allocs_.begin(), allocs_.end(), addr,
            [](uint64_t a, const Range& r) { return a < r.start; });
        if (it == allocs_.begin()) return false;
        --it;
        return addr < it->start + it->len;
    }
};

int main() {
    IovaAllocator a(0x1000, 0x100000);
    auto x = a.alloc(0x4000), y = a.alloc(0x2000);
    std::printf("x=%#lx y=%#lx\n", *x, *y);
    a.free(*x);
    auto z = a.alloc(0x1000);                       // reuses the gap
    std::printf("z=%#lx (gap reuse)\n", *z);
    std::printf("translate %#lx -> %s\n", *y + 0x10,
                a.translate_ok(*y + 0x10) ? "OK" : "FAULT");
}
