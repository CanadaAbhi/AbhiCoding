// vec_bitmap_alloc.cpp (g++ -O2 -std=c++17 vec_bitmap_alloc.cpp)
#include <vector>
#include <cstdint>
#include <cstdio>
#include <optional>

class BitmapAllocator {
    std::vector<uint64_t> bits_;   // 1 bit per page, 1 = allocated
    uint64_t base_, page_size_, num_pages_;
    uint64_t hint_ = 0;            // next-fit scan start

    bool test(uint64_t i) const { return bits_[i >> 6] & (1ull << (i & 63)); }
    void set(uint64_t i)   { bits_[i >> 6] |=  (1ull << (i & 63)); }
    void clear(uint64_t i) { bits_[i >> 6] &= ~(1ull << (i & 63)); }

public:
    BitmapAllocator(uint64_t base, uint64_t size, uint64_t page)
        : bits_((size / page + 63) / 64, 0),
          base_(base), page_size_(page), num_pages_(size / page) {}

    std::optional<uint64_t> alloc(uint64_t bytes) {
        uint64_t need = (bytes + page_size_ - 1) / page_size_;
        for (uint64_t pass = 0; pass < 2; ++pass) {   // next-fit, then wrap
            uint64_t start = pass ? 0 : hint_;
            uint64_t run = 0;
            for (uint64_t i = start; i < num_pages_; ++i) {
                run = test(i) ? 0 : run + 1;
                if (run == need) {
                    uint64_t first = i + 1 - need;
                    for (uint64_t p = first; p <= i; ++p) set(p);
                    hint_ = i + 1;
                    return base_ + first * page_size_;
                }
            }
        }
        return std::nullopt;                           // fragmented/full
    }

    void free(uint64_t addr, uint64_t bytes) {
        uint64_t first = (addr - base_) / page_size_;
        uint64_t need  = (bytes + page_size_ - 1) / page_size_;
        for (uint64_t p = first; p < first + need; ++p) clear(p);
        if (first < hint_) hint_ = first;
    }

    double fragmentation() const {  // free pages not in the largest hole
        uint64_t free_total = 0, best_run = 0, run = 0;
        for (uint64_t i = 0; i < num_pages_; ++i) {
            if (!test(i)) { ++free_total; ++run; if (run > best_run) best_run = run; }
            else run = 0;
        }
        return free_total ? 1.0 - (double)best_run / free_total : 0.0;
    }
};

int main() {
    BitmapAllocator iova(0x8000'0000, 1 << 20, 4096);  // 1MB space, 4K pages
    auto a = iova.alloc(16384);
    auto b = iova.alloc(8192);
    auto c = iova.alloc(16384);
    iova.free(*b, 8192);                   // punch a hole
    auto d = iova.alloc(32768);            // must skip the 8K hole
    std::printf("a=%lx b=%lx c=%lx d=%lx frag=%.2f\n",
                (unsigned long)*a, (unsigned long)*b, (unsigned long)*c,
                (unsigned long)*d, iova.fragmentation());
}
