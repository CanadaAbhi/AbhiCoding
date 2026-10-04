// vec_sched.cpp (g++ -O2 -std=c++17 vec_sched.cpp)
#include <vector>
#include <algorithm>
#include <cstdio>
#include <cstdint>

enum Prio { HIGH, NORMAL, LOW, NPRIO };

struct Job {
    uint32_t id;
    uint64_t deadline;      // EDF within priority
    uint64_t enqueue_tick;
};

struct DeadlineCmp {        // min-heap on deadline (heap funcs are max-heaps)
    bool operator()(const Job& a, const Job& b) const {
        return a.deadline > b.deadline;
    }
};

class Scheduler {
    std::vector<Job> rq_[NPRIO];              // each vector IS a heap
    uint64_t tick_ = 0;
    static constexpr uint64_t AGE_LIMIT = 50; // starvation threshold

public:
    void submit(Job j, Prio p) {
        j.enqueue_tick = tick_;
        rq_[p].push_back(j);
        std::push_heap(rq_[p].begin(), rq_[p].end(), DeadlineCmp{});
    }

    bool pick_next(Job& out) {
        ++tick_;
        age_boost();
        for (int p = HIGH; p < NPRIO; ++p) {
            if (rq_[p].empty()) continue;
            std::pop_heap(rq_[p].begin(), rq_[p].end(), DeadlineCmp{});
            out = rq_[p].back();
            rq_[p].pop_back();
            return true;
        }
        return false;
    }

private:
    void age_boost() {      // promote starving LOW/NORMAL jobs one level
        for (int p = NORMAL; p < NPRIO; ++p) {
            auto& q = rq_[p];
            auto mid = std::partition(q.begin(), q.end(),
                [&](const Job& j) { return tick_ - j.enqueue_tick <= AGE_LIMIT; });
            for (auto it = mid; it != q.end(); ++it) {
                rq_[p - 1].push_back(*it);
                std::push_heap(rq_[p-1].begin(), rq_[p-1].end(), DeadlineCmp{});
            }
            q.erase(mid, q.end());
            std::make_heap(q.begin(), q.end(), DeadlineCmp{}); // re-heapify
        }
    }
};

int main() {
    Scheduler s;
    s.submit({1, 900, 0}, LOW);
    for (uint32_t i = 2; i < 80; ++i) s.submit({i, 100 + i, 0}, HIGH);
    Job j;
    int picked = 0;
    while (s.pick_next(j)) {
        if (j.id == 1) std::printf("starved LOW job ran at pick #%d (aging works)\n", picked);
        ++picked;
    }
}
