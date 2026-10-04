#include <iostream>
#include <unordered_map>
#include <unordered_multimap>
#include <unordered_set>
#include <unordered_multiset>
#include <string>
#include <vector>
#include <iomanip>
#include <algorithm>

// ─────────────────────────────────────────────────
// 1. unordered_map: PID → {priority, state, name}
// ─────────────────────────────────────────────────
struct ProcessInfo {
    std::string name;
    int         priority;   // 0 = highest, 19 = lowest (nice-style)
    std::string state;      // RUNNING, SLEEPING, ZOMBIE, STOPPED
};

void demo_process_table() {
    std::cout << "\n=== [unordered_map] Process Table ===\n";

    std::unordered_map<int, ProcessInfo> proc_table = {
        {1,    {"systemd",    0,  "RUNNING"}},
        {234,  {"kworker",    5,  "SLEEPING"}},
        {512,  {"gpu_drv",    2,  "RUNNING"}},
        {1024, {"sensor_fw", 10,  "SLEEPING"}},
        {2048, {"ota_mgr",    7,  "RUNNING"}},
    };

    // Spawn a new process
    proc_table[3000] = {"user_app", 15, "RUNNING"};
    std::cout << "[SPAWNED] PID 3000 → user_app\n";

    // Context switch: change state
    proc_table[512].state = "SLEEPING";
    proc_table[234].state = "RUNNING";
    std::cout << "[CTX SWITCH] gpu_drv → SLEEPING, kworker → RUNNING\n";

    // Priority boost (anti-starvation)
    for (auto& [pid, info] : proc_table) {
        if (info.state == "SLEEPING" && info.priority > 0) {
            info.priority--;
        }
    }
    std::cout << "[AGING] Priority boosted for sleeping processes\n";

    // Print full process table
    std::cout << "\n" << std::left
              << std::setw(8)  << "PID"
              << std::setw(15) << "NAME"
              << std::setw(10) << "PRIORITY"
              << "STATE\n";
    std::cout << std::string(45, '-') << "\n";
    for (auto& [pid, info] : proc_table) {
        std::cout << std::setw(8)  << pid
                  << std::setw(15) << info.name
                  << std::setw(10) << info.priority
                  << info.state << "\n";
    }

    // Reap zombie processes
    proc_table.erase(1024);
    std::cout << "\n[REAPED] PID 1024 removed.\n";
    std::cout << "[PROC COUNT] " << proc_table.size() << " processes\n";
}

// ─────────────────────────────────────────────────────────
// 2. unordered_multimap: PID → multiple memory regions (VMA)
// ─────────────────────────────────────────────────────────
struct MemRegion {
    std::string name;
    size_t      size_kb;
    std::string perms;  // rwxp
};

void demo_vma_table() {
    std::cout << "\n=== [unordered_multimap] PID → Memory Regions (VMA) ===\n";

    // Key=PID, Value=region description string "name:size_kb:perms"
    std::unordered_multimap<int, std::string> vma_map;

    // PID 512 (gpu_drv) memory layout
    vma_map.emplace(512, "text:256:r-xp");
    vma_map.emplace(512, "data:64:rw-p");
    vma_map.emplace(512, "heap:1024:rw-p");
    vma_map.emplace(512, "gpu_mmio:4096:rw-p");
    vma_map.emplace(512, "dma_buf:8192:rw-p");

    // PID 3000 (user_app) memory layout
    vma_map.emplace(3000, "text:128:r-xp");
    vma_map.emplace(3000, "data:32:rw-p");
    vma_map.emplace(3000, "heap:512:rw-p");
    vma_map.emplace(3000, "stack:256:rw-p");

    // PID 1 (systemd)
    vma_map.emplace(1, "text:512:r-xp");
    vma_map.emplace(1, "heap:2048:rw-p");

    // Print memory map for PID 512
    auto print_vma = [&](int pid) {
        auto [beg, fin] = vma_map.equal_range(pid);
        std::cout << "[VMA MAP for PID " << pid << "]\n";
        size_t total = 0;
        for (auto it = beg; it != fin; ++it) {
            // parse "name:size:perms"
            std::string entry = it->second;
            size_t c1 = entry.find(':');
            size_t c2 = entry.rfind(':');
            std::string name  = entry.substr(0, c1);
            size_t size_kb    = std::stoul(entry.substr(c1+1, c2-c1-1));
            std::string perms = entry.substr(c2+1);
            std::cout << "  " << std::setw(12) << name
                      << std::setw(8) << size_kb << " KB  "
                      << perms << "\n";
            total += size_kb;
        }
        std::cout << "  TOTAL: " << total << " KB ("
                  << vma_map.count(pid) << " regions)\n";
    };

    print_vma(512);
    print_vma(3000);

    // mmap a new anonymous region for PID 3000
    vma_map.emplace(3000, "anon:4096:rw-p");
    std::cout << "\n[MMAP] PID 3000 added anon region.\n";
    std::cout << "[VMA COUNT] PID 3000 now has "
              << vma_map.count(3000) << " regions\n";

    // munmap: remove stack from PID 3000
    auto range = vma_map.equal_range(3000);
    for (auto it = range.first; it != range.second; ++it) {
        if (it->second.find("stack") != std::string::npos) {
            vma_map.erase(it);
            std::cout << "[MUNMAP] PID 3000 stack region removed.\n";
            break;
        }
    }
}

// ─────────────────────────────────────────────────────────
// 3. unordered_set: runnable PID set (scheduler run queue)
// ─────────────────────────────────────────────────────────
void demo_run_queue() {
    std::cout << "\n=== [unordered_set] Scheduler Run Queue ===\n";

    std::unordered_set<int> run_queue;
    std::unordered_set<int> blocked_set;

    // Processes waking up
    std::vector<int> wake_events = {512, 234, 3000, 512, 1, 2048, 234};
    std::cout << "[WAKEUP EVENTS]\n";
    for (int pid : wake_events) {
        if (blocked_set.count(pid)) blocked_set.erase(pid);
        auto [it, ok] = run_queue.insert(pid);
        std::cout << "  PID " << pid
                  << (ok ? " → enqueued" : " → already runnable") << "\n";
    }

    // Scheduler picks next process (simulate round-robin pick)
    std::cout << "\n[SCHEDULE TICK]\n";
    for (int i = 0; i < 3 && !run_queue.empty(); ++i) {
        int picked = *run_queue.begin();
        run_queue.erase(run_queue.begin());
        blocked_set.insert(picked);
        std::cout << "  [RUN] PID " << picked << " scheduled, "
                  << run_queue.size() << " remaining\n";
    }

    // I/O completion → re-enqueue
    std::cout << "\n[I/O COMPLETE] PID 234 unblocked\n";
    blocked_set.erase(234);
    run_queue.insert(234);

    std::cout << "[RUN QUEUE] " << run_queue.size() << " process(es) ready: ";
    for (int pid : run_queue) std::cout << pid << " ";
    std::cout << "\n";
}

// ──────────────────────────────────────────────────────────
// 4. unordered_multiset: CPU event type frequency (perf)
// ──────────────────────────────────────────────────────────
void demo_cpu_events() {
    std::cout << "\n=== [unordered_multiset] CPU Performance Events ===\n";

    // Sampled CPU event stream
    std::unordered_multiset<std::string> perf_events = {
        "cache-miss",   "branch-miss",  "cache-miss",
        "page-fault",   "cache-miss",   "tlb-miss",
        "branch-miss",  "cache-miss",   "ctx-switch",
        "page-fault",   "cache-miss",   "tlb-miss",
        "branch-miss",  "cache-miss",   "ctx-switch",
        "page-fault",   "tlb-miss",     "cache-miss",
        "branch-miss",  "ctx-switch"
    };

    // Build frequency table
    std::unordered_set<std::string> unique(perf_events.begin(),
                                           perf_events.end());
    size_t total = perf_events.size();

    std::cout << std::left
              << std::setw(16) << "Event"
              << std::setw(8)  << "Count"
              << std::setw(10) << "Pct"
              << "Severity\n";
    std::cout << std::string(55, '-') << "\n";

    for (auto& ev : unique) {
        size_t cnt = perf_events.count(ev);
        double pct = 100.0 * cnt / total;
        std::string sev = (pct > 30) ? "CRITICAL"
                        : (pct > 15) ? "HIGH"
                        : (pct > 8)  ? "MEDIUM"
                                     : "LOW";
        std::cout << std::setw(16) << ev
                  << std::setw(8)  << cnt
                  << std::setw(9)  << std::fixed
                  << std::setprecision(1) << pct << "%  "
                  << sev << "\n";
    }

    // Drain all page-faults (cleared after handler runs)
    size_t cleared = perf_events.count("page-fault");
    perf_events.erase("page-fault");
    std::cout << "\n[CLEARED] " << cleared
              << " page-fault events handled.\n";
    std::cout << "[REMAINING EVENTS] " << perf_events.size() << "\n";
}

int main() {
    std::cout << "╔══════════════════════════════════════════╗\n";
    std::cout << "║    Process Scheduler & Memory Tracker    ║\n";
    std::cout << "╚══════════════════════════════════════════╝\n";

    demo_process_table();
    demo_vma_table();
    demo_run_queue();
    demo_cpu_events();

    return 0;
}
