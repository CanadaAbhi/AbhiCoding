#include <iostream>
#include <unordered_map>
#include <unordered_multimap>
#include <unordered_set>
#include <unordered_multiset>
#include <string>
#include <vector>

// ─────────────────────────────────────────────
// 1. unordered_map: driver_name → assigned IRQ
// ─────────────────────────────────────────────
void demo_driver_irq_map() {
    std::cout << "\n=== [unordered_map] Driver → IRQ Assignment ===\n";

    std::unordered_map<std::string, int> driver_irq = {
        {"i2c_drv",    42},
        {"spi_drv",    17},
        {"uart_drv",   33},
        {"gpu_drv",    64},
        {"pcie_drv",   88}
    };

    // Register a new driver
    driver_irq["dma_drv"] = 55;

    // Look up IRQ for a specific driver
    std::string query = "spi_drv";
    auto it = driver_irq.find(query);
    if (it != driver_irq.end()) {
        std::cout << "[FOUND] " << query << " → IRQ " << it->second << "\n";
    }

    // Update IRQ (driver re-assigned)
    driver_irq["uart_drv"] = 99;
    std::cout << "[UPDATED] uart_drv → IRQ " << driver_irq["uart_drv"] << "\n";

    // Detect collision: two drivers requesting same IRQ
    std::unordered_map<int, std::string> irq_owner;
    bool collision = false;
    for (auto& [drv, irq] : driver_irq) {
        if (irq_owner.count(irq)) {
            std::cout << "[COLLISION] IRQ " << irq
                      << " claimed by both '" << irq_owner[irq]
                      << "' and '" << drv << "'\n";
            collision = true;
        } else {
            irq_owner[irq] = drv;
        }
    }
    if (!collision) std::cout << "[OK] No IRQ collisions detected.\n";

    // Print all registered drivers
    std::cout << "\n[ALL DRIVERS]\n";
    for (auto& [drv, irq] : driver_irq) {
        std::cout << "  " << drv << " → IRQ " << irq << "\n";
    }
}

// ────────────────────────────────────────────────────────────
// 2. unordered_multimap: bus_name → multiple device endpoints
// ────────────────────────────────────────────────────────────
void demo_bus_device_multimap() {
    std::cout << "\n=== [unordered_multimap] Bus → Device Endpoints ===\n";

    std::unordered_multimap<std::string, std::string> bus_devices;

    // Multiple devices on the same I2C bus
    bus_devices.emplace("i2c-0", "temp_sensor@0x48");
    bus_devices.emplace("i2c-0", "eeprom@0x50");
    bus_devices.emplace("i2c-0", "imu@0x68");

    // Multiple devices on SPI bus
    bus_devices.emplace("spi-1", "flash@CS0");
    bus_devices.emplace("spi-1", "display@CS1");

    // USB bus
    bus_devices.emplace("usb-2", "camera@port1");
    bus_devices.emplace("usb-2", "keyboard@port2");
    bus_devices.emplace("usb-2", "hub@port3");

    // Query all devices on i2c-0
    std::string bus = "i2c-0";
    auto [begin, end] = bus_devices.equal_range(bus);
    std::cout << "[DEVICES on " << bus << "]\n";
    for (auto it = begin; it != end; ++it) {
        std::cout << "  → " << it->second << "\n";
    }

    // Count devices per bus
    std::cout << "\n[DEVICE COUNT per BUS]\n";
    std::unordered_set<std::string> seen_buses;
    for (auto& [bus_name, dev] : bus_devices) {
        if (seen_buses.insert(bus_name).second) {
            std::cout << "  " << bus_name
                      << " → " << bus_devices.count(bus_name) << " device(s)\n";
        }
    }

    // Remove all devices from spi-1 (simulate bus reset)
    std::cout << "\n[REMOVING all devices from spi-1]\n";
    bus_devices.erase("spi-1");
    std::cout << "  spi-1 device count after reset: "
              << bus_devices.count("spi-1") << "\n";
}

// ─────────────────────────────────────────────────────────
// 3. unordered_set: unique active device IDs in the system
// ─────────────────────────────────────────────────────────
void demo_active_device_set() {
    std::cout << "\n=== [unordered_set] Unique Active Device IDs ===\n";

    std::unordered_set<std::string> active_devices;

    // Devices coming online
    std::vector<std::string> events = {
        "dev:0001", "dev:0002", "dev:0003",
        "dev:0001",  // duplicate — already active
        "dev:0004",
        "dev:0002",  // duplicate
        "dev:0005"
    };

    std::cout << "[REGISTRATION LOG]\n";
    for (auto& dev : events) {
        auto [it, inserted] = active_devices.insert(dev);
        if (inserted) {
            std::cout << "  [ONLINE]  " << dev << "\n";
        } else {
            std::cout << "  [SKIP]    " << dev << " already active\n";
        }
    }

    // Device going offline
    std::string offline = "dev:0003";
    active_devices.erase(offline);
    std::cout << "\n[OFFLINE] " << offline << " removed.\n";

    // Check membership
    std::cout << "\n[MEMBERSHIP CHECK]\n";
    for (auto& id : {"dev:0001", "dev:0003", "dev:0005"}) {
        std::cout << "  " << id << " → "
                  << (active_devices.count(id) ? "ACTIVE" : "OFFLINE") << "\n";
    }

    std::cout << "\n[TOTAL ACTIVE] " << active_devices.size() << " device(s)\n";
}

// ────────────────────────────────────────────────────────────
// 4. unordered_multiset: duplicate interrupt event frequency
// ────────────────────────────────────────────────────────────
void demo_interrupt_event_multiset() {
    std::cout << "\n=== [unordered_multiset] Interrupt Event Frequency ===\n";

    // Simulated IRQ event stream (IRQ line numbers)
    std::unordered_multiset<int> irq_events = {
        42, 17, 42, 64, 17, 42, 33, 88,
        42, 64, 17, 88, 42, 33, 64, 88,
        17, 42, 88, 33
    };

    // Frequency analysis per IRQ line
    std::unordered_set<int> seen;
    std::cout << "[IRQ FREQUENCY TABLE]\n";
    std::cout << "  IRQ\t| Count\t| Status\n";
    std::cout << "  ────────────────────────\n";
    for (int irq : irq_events) {
        if (seen.insert(irq).second) {
            size_t freq = irq_events.count(irq);
            std::string status = (freq >= 5) ? "⚠ HIGH LOAD"
                               : (freq >= 3) ? "MODERATE"
                                             : "normal";
            std::cout << "  IRQ " << irq << "\t| " << freq
                      << "\t| " << status << "\n";
        }
    }

    // Most-fired IRQ
    int max_irq = -1;
    size_t max_count = 0;
    for (int irq : seen) {
        if (irq_events.count(irq) > max_count) {
            max_count = irq_events.count(irq);
            max_irq = irq;
        }
    }
    std::cout << "\n[HOTTEST IRQ] IRQ " << max_irq
              << " fired " << max_count << " times\n";

    // Drain one occurrence of IRQ 42 (handled)
    auto it = irq_events.find(42);
    if (it != irq_events.end()) irq_events.erase(it);
    std::cout << "[HANDLED] IRQ 42 — remaining: "
              << irq_events.count(42) << "\n";

    std::cout << "[TOTAL EVENTS in queue] " << irq_events.size() << "\n";
}

int main() {
    std::cout << "╔══════════════════════════════════════════╗\n";
    std::cout << "║    Driver Registry & IRQ Tracker Lab     ║\n";
    std::cout << "╚══════════════════════════════════════════╝\n";

    demo_driver_irq_map();
    demo_bus_device_multimap();
    demo_active_device_set();
    demo_interrupt_event_multiset();

    return 0;
}
