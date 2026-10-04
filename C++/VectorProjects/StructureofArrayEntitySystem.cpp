// vec_ecs.cpp (g++ -O3 -march=native -std=c++17 vec_ecs.cpp)
#include <vector>
#include <chrono>
#include <cstdio>
#include <random>
#include <cstdint>

// SoA: each field is its own contiguous vector -> SIMD-friendly
struct Particles {
    std::vector<float> x, y, vx, vy;
    std::vector<float> life;
    std::vector<uint32_t> id;

    size_t size() const { return x.size(); }

    void spawn(uint32_t i, std::mt19937& rng) {
        std::uniform_real_distribution<float> u(-1, 1);
        x.push_back(0); y.push_back(0);
        vx.push_back(u(rng)); vy.push_back(u(rng));
        life.push_back(1.0f + u(rng));
        id.push_back(i);
    }

    void kill(size_t i) {               // swap-and-pop: O(1), stays dense
        size_t last = size() - 1;
        x[i] = x[last];   y[i] = y[last];
        vx[i] = vx[last]; vy[i] = vy[last];
        life[i] = life[last]; id[i] = id[last];
        x.pop_back(); y.pop_back(); vx.pop_back();
        vy.pop_back(); life.pop_back(); id.pop_back();
    }

    void update(float dt) {
        size_t n = size();
        for (size_t i = 0; i < n; ++i) { // tight loops auto-vectorize
            x[i] += vx[i] * dt;
            y[i] += vy[i] * dt;
            life[i] -= dt;
        }
        for (size_t i = 0; i < size(); )
            (life[i] <= 0) ? kill(i) : (void)++i;
    }
};

// AoS comparison baseline
struct ParticleAoS { float x, y, vx, vy, life; uint32_t id; char pad[40]; };

int main() {
    std::mt19937 rng(1);
    Particles soa;
    std::vector<ParticleAoS> aos;
    for (uint32_t i = 0; i < 2'000'000; ++i) {
        soa.spawn(i, rng);
        aos.push_back({soa.x[i], soa.y[i], soa.vx[i], soa.vy[i], soa.life[i], i, {}});
    }

    auto t0 = std::chrono::steady_clock::now();
    for (int f = 0; f < 60; ++f) soa.update(0.016f);
    auto soa_ms = std::chrono::duration<double, std::milli>(
                      std::chrono::steady_clock::now() - t0).count();

    t0 = std::chrono::steady_clock::now();
    for (int f = 0; f < 60; ++f)
        for (auto& p : aos) { p.x += p.vx * .016f; p.y += p.vy * .016f; p.life -= .016f; }
    auto aos_ms = std::chrono::duration<double, std::milli>(
                      std::chrono::steady_clock::now() - t0).count();

    std::printf("SoA: %.1f ms   AoS(padded): %.1f ms   speedup %.2fx\n",
                soa_ms, aos_ms, aos_ms / soa_ms);
}
