#include <iostream>
#include <unordered_map>
#include <unordered_multimap>
#include <unordered_set>
#include <unordered_multiset>
#include <string>
#include <vector>
#include <iomanip>

// ─────────────────────────────────────────────
// 1. unordered_map: flow_key → forwarding_port
// ─────────────────────────────────────────────
void demo_flow_table() {
    std::cout << "\n=== [unordered_map] Flow Table: flow_key → port ===\n";

    // Flow key = "src_ip:dst_ip:proto"
    std::unordered_map<std::string, int> flow_table;

    // Install flow rules
    flow_table["192.168.1.1:10.0.0.1:TCP"] = 1;
    flow_table["192.168.1.2:10.0.0.2:UDP"] = 2;
    flow_table["10.0.0.5:172.16.0.1:ICMP"] = 3;
    flow_table["192.168.2.1:8.8.8.8:TCP"]  = 4;

    // Incoming packet lookup
    std::vector<std::string> packets = {
        "192.168.1.1:10.0.0.1:TCP",
        "10.0.0.5:172.16.0.1:ICMP",
        "192.168.9.9:1.1.1.1:UDP"   // unknown flow → drop
    };

    std::cout << std::left << std::setw(35) << "Flow Key"
              << "Action\n";
    std::cout << std::string(50, '-') << "\n";

    for (auto& pkt : packets) {
        auto it = flow_table.find(pkt);
        if (it != flow_table.end()) {
            std::cout << std::setw(35) << pkt
                      << "FORWARD → port " << it->second << "\n";
        } else {
            std::cout << std::setw(35) << pkt
                      << "DROP (no rule)\n";
        }
    }

    // Update a rule (traffic engineering)
    flow_table["192.168.1.1:10.0.0.1:TCP"] = 2;
    std::cout << "\n[RULE UPDATE] 192.168.1.1:10.0.0.1:TCP → port "
              << flow_table["192.168.1.1:10.0.0.1:TCP"] << "\n";

    // Remove an expired flow
    flow_table.erase("10.0.0.5:172.16.0.1:ICMP");
    std::cout << "[EXPIRED]    10.0.0.5:172.16.0.1:ICMP removed. "
              << "Rules remaining: " << flow_table.size() << "\n";
}

// ───────────────────────────────────────────────────────
// 2. unordered_multimap: destination → multiple next-hops
// ───────────────────────────────────────────────────────
void demo_multipath_routes() {
    std::cout << "\n=== [unordered_multimap] Dest → Multi-path Next Hops ===\n";

    std::unordered_multimap<std::string, std::string> routes;

    // ECMP (Equal-Cost Multi-Path) routing
    routes.emplace("10.0.0.0/24", "via 192.168.1.1 dev eth0");
    routes.emplace("10.0.0.0/24", "via 192.168.1.2 dev eth1");
    routes.emplace("10.0.0.0/24", "via 192.168.1.3 dev eth2");

    routes.emplace("172.16.0.0/16", "via 10.1.1.1 dev eth0");
    routes.emplace("172.16.0.0/16", "via 10.1.1.2 dev eth1");

    routes.emplace("0.0.0.0/0", "via 203.0.113.1 dev wan0");  // default

    // Show all paths to a destination
    auto show_routes = [&](const std::string& dest) {
        auto [beg, fin] = routes.equal_range(dest);
        std::cout << "[ROUTES to " << dest << "]\n";
        int hop = 1;
        for (auto it = beg; it != fin; ++it, ++hop) {
            std::cout << "  Path " << hop << ": " << it->second << "\n";
        }
        std::cout << "  Total paths: " << routes.count(dest) << "\n";
    };

    show_routes("10.0.0.0/24");
    show_routes("172.16.0.0/16");
    show_routes("0.0.0.0/0");

    // Remove one failed path
    std::cout << "\n[FAILOVER] Removing eth1 path to 10.0.0.0/24\n";
    auto range = routes.equal_range("10.0.0.0/24");
    for (auto it = range.first; it != range.second; ++it) {
        if (it->second.find("eth1") != std::string::npos) {
            routes.erase(it);
            break;
        }
    }
    show_routes("10.0.0.0/24");
}

// ───────────────────────────────────────────────────────
// 3. unordered_set: unique learned MAC addresses (L2 CAM)
// ───────────────────────────────────────────────────────
void demo_mac_table() {
    std::cout << "\n=== [unordered_set] Learned MAC Address Table ===\n";

    std::unordered_set<std::string> mac_table;

    // Frames arriving with source MACs
    std::vector<std::string> src_macs = {
        "AA:BB:CC:DD:EE:01",
        "AA:BB:CC:DD:EE:02",
        "AA:BB:CC:DD:EE:01",  // seen again
        "AA:BB:CC:DD:EE:03",
        "DE:AD:BE:EF:00:01",
        "AA:BB:CC:DD:EE:02",  // seen again
        "DE:AD:BE:EF:00:02",
        "FF:FF:FF:FF:FF:FF"   // broadcast
    };

    std::cout << "[MAC LEARNING LOG]\n";
    for (auto& mac : src_macs) {
        auto [it, learned] = mac_table.insert(mac);
        if (learned) {
            std::cout << "  [LEARNED]  " << mac << "\n";
        } else {
            std::cout << "  [REFRESH]  " << mac << " (already known)\n";
        }
    }

    // Lookup for forwarding decision
    std::cout << "\n[FORWARDING DECISION]\n";
    std::vector<std::string> dst_macs = {
        "AA:BB:CC:DD:EE:03",  // known → unicast
        "11:22:33:44:55:66",  // unknown → flood
        "FF:FF:FF:FF:FF:FF"   // broadcast → flood
    };
    for (auto& mac : dst_macs) {
        bool known = mac_table.count(mac) > 0;
        bool is_broadcast = (mac == "FF:FF:FF:FF:FF:FF");
        std::cout << "  DST " << mac << " → "
                  << (is_broadcast ? "FLOOD (broadcast)"
                     : known       ? "UNICAST (known)"
                                   : "FLOOD (unknown)") << "\n";
    }

    // Age out one entry
    mac_table.erase("DE:AD:BE:EF:00:01");
    std::cout << "\n[AGED OUT] DE:AD:BE:EF:00:01\n";
    std::cout << "[CAM TABLE SIZE] " << mac_table.size() << " entries\n";
}

// ──────────────────────────────────────────────────────────
// 4. unordered_multiset: protocol event frequency analysis
// ──────────────────────────────────────────────────────────
void demo_protocol_events() {
    std::cout << "\n=== [unordered_multiset] Protocol Event Counter ===\n";

    // Simulated protocol event stream
    std::unordered_multiset<std::string> events = {
        "TCP_SYN",  "TCP_SYN",  "TCP_SYN",  "TCP_ACK",
        "TCP_ACK",  "TCP_FIN",  "UDP_PKT",  "UDP_PKT",
        "ICMP_REQ", "ARP_REQ",  "TCP_SYN",  "UDP_PKT",
        "TCP_RST",  "ICMP_REP", "ARP_REP",  "TCP_SYN",
        "TCP_ACK",  "UDP_PKT",  "TCP_RST",  "ICMP_REQ"
    };

    // Frequency table
    std::unordered_set<std::string> unique_events(events.begin(), events.end());

    std::cout << std::left << std::setw(15) << "Event"
              << std::setw(10) << "Count"
              << "Bar\n";
    std::cout << std::string(50, '-') << "\n";

    std::string top_event;
    size_t top_count = 0;

    for (auto& ev : unique_events) {
        size_t cnt = events.count(ev);
        std::string bar(cnt * 2, '#');
        std::cout << std::setw(15) << ev
                  << std::setw(10) << cnt
                  << bar << "\n";
        if (cnt > top_count) { top_count = cnt; top_event = ev; }
    }

    std::cout << "\n[TOP EVENT] " << top_event
              << " (" << top_count << " occurrences)\n";

    // Detect SYN flood (> 3 TCP_SYN within window)
    if (events.count("TCP_SYN") > 3) {
        std::cout << "[ALERT] Possible SYN flood detected! ("
                  << events.count("TCP_SYN") << " SYNs)\n";
    }

    // Remove all TCP_RST events (filtered)
    events.erase("TCP_RST");
    std::cout << "[FILTERED] TCP_RST removed. "
              << "Queue size: " << events.size() << "\n";
}

int main() {
    std::cout << "╔══════════════════════════════════════════╗\n";
    std::cout << "║   Network Packet Classifier & Flow Lab   ║\n";
    std::cout << "╚══════════════════════════════════════════╝\n";

    demo_flow_table();
    demo_multipath_routes();
    demo_mac_table();
    demo_protocol_events();

    return 0;
}
