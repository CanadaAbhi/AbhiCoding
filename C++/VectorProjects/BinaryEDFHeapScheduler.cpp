// vec_heap_sched.cpp (g++ -O2 -std=c++17 vec_heap_sched.cpp)
#include <vector>
#include <cstdint>
#include <cstdio>

struct Job {
    uint64_t id;
    int      priority;     // 0 = HIGH, 1 = NORMAL, 2 = LOW
    uint64_t deadline_us;  // EDF within same priority
    bool operator<(const Job& o) const {   // "more urgent than"
        if (priority != o.priority) return priority < o.priority;
        return deadline_us < o.deadline_us;
    }
};

class JobHeap {
    std::vector<Job> h_;   // implicit binary tree: children of i at 2i+1, 2i+2

    void sift_up(std::size_t i) {
        while (i > 0) {
            std::size_t p = (i - 1) / 2;
            if (!(h_[i] < h_[p])) break;
            std::swap(h_[i], h_[p]);
            i = p;
        }
    }
    void sift_down(std::size_t i) {
        for (;;) {
            std::size_t l = 2 * i + 1, r = l + 1, best = i;
            if (l < h_.size() && h_[l] < h_[best]) best = l;
            if (r < h_.size() && h_[r] < h_[best]) best = r;
            if (best == i) break;
            std::swap(h_[i], h_[best]);
            i = best;
        }
    }

public:
    void push(Job j) { h_.push_back(j); sift_up(h_.size() - 1); }
    Job pop() {
        Job top = h_[0];
        h_[0] = h_.back();
        h_.pop_back();
        if (!h_.empty()) sift_down(0);
        return top;
    }
    bool empty() const { return h_.empty(); }

    // Aging: anti-starvation pass, like your drm_sched project
    void age_all(uint64_t now_us, uint64_t threshold_us) {
        bool changed = false;
        for (auto& j : h_)
            if (j.priority > 0 && now_us > j.deadline_us + threshold_us) {
                j.priority--;              // promote starving jobs
                changed = true;
            }
        if (changed)                       // re-heapify in O(n)
            for (std::size_t i = h_.size() / 2; i-- > 0;) sift_down(i);
    }
};

int main() {
    JobHeap q;
    q.push({1, 2, 100});   // LOW, early deadline
    q.push({2, 0, 900});   // HIGH, late deadline
    q.push({3, 0, 300});   // HIGH, early deadline
    q.push({4, 1, 200});   // NORMAL

    q.age_all(/*now=*/5000, /*threshold=*/1000);  // job 1 gets promoted

    while (!q.empty()) {
        Job j = q.pop();
        std::printf("dispatch job %lu (prio=%d deadline=%lu)\n",
                    (unsigned long)j.id, j.priority,
                    (unsigned long)j.deadline_us);
    }
}
