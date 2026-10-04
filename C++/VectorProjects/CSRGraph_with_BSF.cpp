// vec_graph.cpp (g++ -O2 -std=c++17 vec_graph.cpp)
#include <vector>
#include <queue>
#include <cstdio>
#include <cstdint>

// CSR: offsets[v]..offsets[v+1] indexes into edges[] — zero pointer chasing
struct Csr {
    std::vector<uint32_t> offsets;   // size V+1
    std::vector<uint32_t> edges;     // size E

    static Csr build(uint32_t v, const std::vector<std::pair<uint32_t,uint32_t>>& el) {
        Csr g;
        g.offsets.assign(v + 1, 0);
        for (auto& e : el) g.offsets[e.first + 1]++;
        for (uint32_t i = 1; i <= v; ++i) g.offsets[i] += g.offsets[i - 1];
        g.edges.resize(el.size());
        std::vector<uint32_t> cur(g.offsets.begin(), g.offsets.end() - 1);
        for (auto& e : el) g.edges[cur[e.first]++] = e.second;
        return g;
    }
    uint32_t nverts() const { return (uint32_t)offsets.size() - 1; }
};

// Kahn's algorithm: exactly how lpu_sim resolves deps at "compile time"
std::vector<uint32_t> topo_sort(const Csr& g) {
    std::vector<uint32_t> indeg(g.nverts(), 0), order;
    for (uint32_t e : g.edges) indeg[e]++;
    std::vector<uint32_t> ready;                 // vector as worklist
    for (uint32_t v = 0; v < g.nverts(); ++v)
        if (!indeg[v]) ready.push_back(v);
    while (!ready.empty()) {
        uint32_t v = ready.back(); ready.pop_back();
        order.push_back(v);
        for (uint32_t i = g.offsets[v]; i < g.offsets[v + 1]; ++i)
            if (--indeg[g.edges[i]] == 0) ready.push_back(g.edges[i]);
    }
    if (order.size() != g.nverts()) order.clear();  // cycle detected
    return order;
}

int main() {
    // Job dependency DAG: 0->2, 1->2, 2->3, 2->4, 3->5, 4->5
    auto g = Csr::build(6, {{0,2},{1,2},{2,3},{2,4},{3,5},{4,5}});
    auto order = topo_sort(g);
    if (order.empty()) { std::puts("cycle!"); return 1; }
    std::printf("static schedule: ");
    for (uint32_t v : order) std::printf("%u ", v);
    std::puts("");
}
