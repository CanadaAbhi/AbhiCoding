// vec_arena.cpp (g++ -O2 -std=c++17 vec_arena.cpp)
#include <vector>
#include <cstdint>
#include <cstdio>
#include <cassert>

// Handle-based pool: objects live contiguously in a std::vector,
// freed slots are chained through a free-list (like a GEM handle table).
template <typename T>
class Pool {
    struct Slot {
        alignas(T) unsigned char storage[sizeof(T)];
        uint32_t next_free;      // index chain; UINT32_MAX = occupied
        uint32_t generation;     // detects stale handles (use-after-free)
    };
    std::vector<Slot> slots_;
    uint32_t          free_head_ = UINT32_MAX;

public:
    struct Handle { uint32_t index; uint32_t gen; };

    template <typename... Args>
    Handle create(Args&&... args) {
        uint32_t idx;
        if (free_head_ != UINT32_MAX) {
            idx = free_head_;
            free_head_ = slots_[idx].next_free;
        } else {
            idx = (uint32_t)slots_.size();
            slots_.emplace_back();
            slots_.back().generation = 0;
        }
        Slot& s = slots_[idx];
        s.next_free = UINT32_MAX;
        new (s.storage) T(std::forward<Args>(args)...);
        return {idx, s.generation};
    }

    T* get(Handle h) {
        if (h.index >= slots_.size()) return nullptr;
        Slot& s = slots_[h.index];
        if (s.next_free != UINT32_MAX || s.generation != h.gen)
            return nullptr;      // freed or stale handle
        return reinterpret_cast<T*>(s.storage);
    }

    void destroy(Handle h) {
        T* obj = get(h);
        assert(obj && "double free or stale handle");
        obj->~T();
        Slot& s = slots_[h.index];
        s.generation++;          // invalidate outstanding handles
        s.next_free = free_head_;
        free_head_  = h.index;
    }
};

struct BufferObject { uint64_t gpu_addr; uint32_t size; };

int main() {
    Pool<BufferObject> pool;
    auto h1 = pool.create(BufferObject{0x1000, 4096});
    auto h2 = pool.create(BufferObject{0x2000, 8192});
    pool.destroy(h1);
    std::printf("h1 after free: %p (expect null)\n", (void*)pool.get(h1));
    std::printf("h2 size: %u\n", pool.get(h2)->size);
    auto h3 = pool.create(BufferObject{0x3000, 1024}); // reuses h1's slot
    std::printf("h3 reused index %u, gen %u\n", h3.index, h3.gen);
}
