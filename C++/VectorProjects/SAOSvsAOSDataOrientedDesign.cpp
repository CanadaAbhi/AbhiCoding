// vec_ecs.cpp (g++ -O3 -march=native -std=c++17 vec_ecs.cpp)
#include <vector>
#include <chrono>
#include <cstdio>
#include <random>

constexpr std::size_t N = 2'000'000;

// --- AoS: one vector of fat structs ---------------------------------
struct EntityAoS {
    float x, y, z;        // position (hot)
    float vx, vy, vz;     // velocity (hot)
    char  name[64];       // cold data dragged into cache anyway
    int   health, level, flags;
};

// --- SoA: parallel vectors, same index = same entity ----------------
struct EntitiesSoA {
    std::vector<float> x, y, z, vx, vy, vz;   // hot arrays only
    std::vector<int>   health;                 // cold, untouched here
    void resize(std::size_t n) {
        x.resize(n); y.resize(n); z.resize(n);
        vx.resize(n); vy.resize(n); vz.resize(n);
        health.resize(n);
    }
};

template <typename F>
double bench(const char* name, F f) {
    auto t0 = std::chrono::steady_clock::now();
    f();
    double ms = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - t0).count();
    std::printf("%-12s %8.2f ms\n", name, ms);
    return ms;
}

int main() {
    std::vector<EntityAoS> aos(N);
    EntitiesSoA soa; soa.resize(N);
    const float dt = 0.016f;

    double t_aos = bench("AoS update", [&] {
        for (auto& e : aos) {           // pulls 100+ bytes per entity
            e.x += e.vx * dt;
            e.y += e.vy * dt;
            e.z += e.vz * dt;
        }
    });
    double t_soa = bench("SoA update", [&] {
        for (std::size_t i = 0; i < N; ++i) {  // pure unit-stride float
            soa.x[i] += soa.vx[i] * dt;        // streams; auto-vectorizes
            soa.y[i] += soa.vy[i] * dt;
            soa.z[i] += soa.vz[i] * dt;
        }
    });
    std::printf("SoA speedup: %.2fx\n", t_aos / t_soa);
}
