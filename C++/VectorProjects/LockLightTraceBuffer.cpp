// vec_flight_recorder.cpp (g++ -O2 -std=c++17 vec_flight_recorder.cpp -lpthread)
#include <vector>
#include <algorithm>
#include <chrono>
#include <thread>
#include <cstdio>

enum class Ev : uint8_t { SubmitEnter, RingPush, JobStart, JobDone, FenceSignal };

struct Event { uint64_t ts_ns; uint64_t job_id; Ev type; uint8_t tid; };

// Per-thread vector = no locking on the hot path (like per-CPU buffers;
// your BPF project used ONE global ringbuf for ordering — merge-sort here
// solves the same cross-CPU ordering problem after the fact).
struct Recorder {
    std::vector<std::vector<Event>> per_thread;
    explicit Recorder(unsigned n) : per_thread(n) {
        for (auto& v : per_thread) v.reserve(1 << 16);  // no realloc in hot path
    }
    void log(unsigned tid, uint64_t job, Ev e) {
        per_thread[tid].push_back({now(), job, e, (uint8_t)tid});
    }
    static uint64_t now() {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
    }
    std::vector<Event> merged() const {       // k-way merge via sort
        std::vector<Event> all;
        for (auto& v : per_thread) all.insert(all.end(), v.begin(), v.end());
        std::stable_sort(all.begin(), all.end(),
            [](const Event& a, const Event& b) { return a.ts_ns < b.ts_ns; });
        return all;
    }
};

int main() {
    Recorder rec(2);

    std::thread producer([&] {               // "driver" thread
        for (uint64_t j = 1; j <= 1000; ++j) {
            rec.log(0, j, Ev::SubmitEnter);
            rec.log(0, j, Ev::RingPush);
        }
    });
    std::thread engine([&] {                  // "GPU engine" thread
        for (uint64_t j = 1; j <= 1000; ++j) {
            rec.log(1, j, Ev::JobStart);
            rec.log(1, j, Ev::JobDone);
            rec.log(1, j, Ev::FenceSignal);
        }
    });
    producer.join(); engine.join();

    auto all = rec.merged();

    // Correlate: per-job submit->fence latency via sorted scan
    std::vector<uint64_t> submit(1001, 0), lat;
    for (auto& e : all) {
        if (e.type == Ev::SubmitEnter) submit[e.job_id] = e.ts_ns;
        if (e.type == Ev::FenceSignal && submit[e.job_id])
            lat.push_back(e.ts_ns - submit[e.job_id]);
    }
    std::sort(lat.begin(), lat.end());
    std::printf("jobs=%zu  P50=%lu ns  P99=%lu ns\n", lat.size(),
                (unsigned long)lat[lat.size() / 2],
                (unsigned long)lat[lat.size() * 99 / 100]);
}
