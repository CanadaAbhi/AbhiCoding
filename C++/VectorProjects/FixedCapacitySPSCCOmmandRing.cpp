// vec_ring.cpp  (g++ -O2 -std=c++17 vec_ring.cpp -lpthread)
#include <vector>
#include <atomic>
#include <thread>
#include <cstdint>
#include <cstdio>

struct Cmd { uint32_t opcode; uint32_t payload; uint64_t fence_id; };

class SpscRing {
    std::vector<Cmd>      buf_;          // capacity fixed at construction
    const std::size_t     mask_;
    std::atomic<uint64_t> head_{0};      // producer writes
    std::atomic<uint64_t> tail_{0};      // consumer writes
public:
    explicit SpscRing(std::size_t pow2_cap)
        : buf_(pow2_cap), mask_(pow2_cap - 1) {}

    bool push(const Cmd& c) {            // producer thread only
        uint64_t h = head_.load(std::memory_order_relaxed);
        if (h - tail_.load(std::memory_order_acquire) == buf_.size())
            return false;                // full
        buf_[h & mask_] = c;
        head_.store(h + 1, std::memory_order_release);  // publish
        return true;
    }
    bool pop(Cmd& out) {                 // consumer thread only
        uint64_t t = tail_.load(std::memory_order_relaxed);
        if (t == head_.load(std::memory_order_acquire))
            return false;                // empty
        out = buf_[t & mask_];
        tail_.store(t + 1, std::memory_order_release);
        return true;
    }
};

int main() {
    SpscRing ring(1024);
    std::atomic<uint64_t> last_fence{0};
    constexpr uint64_t N = 5'000'000;

    std::thread gpu([&] {                // "GPU engine" consumer
        Cmd c;
        uint64_t done = 0;
        while (done < N)
            if (ring.pop(c)) { last_fence.store(c.fence_id); ++done; }
    });
    for (uint64_t i = 1; i <= N; )       // "driver" producer
        if (ring.push({0xC0DE, 42, i})) ++i;

    gpu.join();
    std::printf("completed, last fence signaled = %lu\n",
                (unsigned long)last_fence.load());
}
