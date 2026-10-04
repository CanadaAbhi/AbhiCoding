// vec_telemetry.cpp (g++ -O2 -std=c++17 vec_telemetry.cpp)
#include <vector>
#include <algorithm>
#include <numeric>
#include <cstdio>
#include <random>
#include <cmath>

struct Sample { uint64_t ts_us; double value; };

class Window {                      // fixed-size sliding window
    std::vector<Sample> buf_;
    std::size_t cap_, next_ = 0;
    bool full_ = false;
public:
    explicit Window(std::size_t cap) : cap_(cap) { buf_.reserve(cap); }
    void push(Sample s) {
        if (buf_.size() < cap_) buf_.push_back(s);
        else { buf_[next_] = s; full_ = true; }
        next_ = (next_ + 1) % cap_;
    }
    std::vector<double> values() const {
        std::vector<double> v(buf_.size());
        std::transform(buf_.begin(), buf_.end(), v.begin(),
                       [](const Sample& s) { return s.value; });
        return v;
    }
};

double percentile(std::vector<double> v, double p) {  // copy by design
    if (v.empty()) return 0;
    std::size_t k = (std::size_t)(p / 100.0 * (v.size() - 1));
    std::nth_element(v.begin(), v.begin() + k, v.end());  // O(n), not sort
    return v[k];
}

struct Stats { double mean, stddev, p50, p95, p99; std::size_t outliers; };

Stats analyze(const std::vector<double>& v) {
    Stats s{};
    double sum = std::accumulate(v.begin(), v.end(), 0.0);
    s.mean = sum / v.size();
    double sq = 0;
    for (double x : v) sq += (x - s.mean) * (x - s.mean);
    s.stddev = std::sqrt(sq / v.size());
    s.p50 = percentile(v, 50);
    s.p95 = percentile(v, 95);
    s.p99 = percentile(v, 99);
    s.outliers = std::count_if(v.begin(), v.end(),
        [&](double x) { return std::abs(x - s.mean) > 3 * s.stddev; });
    return s;
}

int main() {
    Window win(10'000);
    std::mt19937 rng(7);
    std::normal_distribution<double> temp(45.0, 2.0);    // ~45C sensor
    std::uniform_real_distribution<double> glitch(90, 120);

    for (uint64_t t = 0; t < 10'000; ++t) {
        double v = (t % 997 == 0) ? glitch(rng) : temp(rng); // inject spikes
        win.push({t * 1000, v});
    }
    Stats s = analyze(win.values());
    std::printf("mean=%.2f stddev=%.2f\n", s.mean, s.stddev);
    std::printf("P50=%.2f P95=%.2f P99=%.2f outliers=%zu\n",
                s.p50, s.p95, s.p99, s.outliers);
}
