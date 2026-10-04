#include <iostream>
#include <unordered_map>
#include <unordered_multimap>
#include <unordered_set>
#include <unordered_multiset>
#include <string>
#include <vector>
#include <iomanip>
#include <sstream>

// ─────────────────────────────────────────────────────
// 1. unordered_map: component → current firmware version
// ─────────────────────────────────────────────────────
struct FirmwareVersion {
    int     major, minor, patch;
    std::string hash;  // SHA-256 short form
    std::string status; // STABLE, BETA, CRITICAL_FIX
};

std::string ver_str(const FirmwareVersion& v) {
    return std::to_string(v.major) + "." +
           std::to_string(v.minor) + "." +
           std::to_string(v.patch);
}

void demo_firmware_map() {
    std::cout << "\n=== [unordered_map] Firmware Version Registry ===\n";

    std::unordered_map<std::string, FirmwareVersion> fw_registry = {
        {"bootloader",  {2, 1, 0, "a1b2c3d4", "STABLE"}},
        {"kernel",      {5, 15, 42, "e5f6a7b8", "STABLE"}},
        {"gpu_drv",     {1, 4, 3,  "c9d0e1f2", "BETA"}},
        {"sensor_fw",   {3, 0, 1,  "a3b4c5d6", "STABLE"}},
        {"ota_manager", {1, 2, 0,  "f7e8d9c0", "CRITICAL_FIX"}},
    };

    // OTA: update kernel to new version
    fw_registry["kernel"] = {5, 16, 0, "11223344", "STABLE"};
    std::cout << "[OTA UPDATE] kernel → v5.16.0\n";

    // Add new component
    fw_registry["wifi_fw"] = {4, 0, 0, "aabbccdd", "BETA"};
    std::cout << "[NEW COMPONENT] wifi_fw v4.0.0 registered\n";

    // Find all CRITICAL_FIX components
    std::cout << "\n[CRITICAL COMPONENTS]\n";
    for (auto& [comp, fw] : fw_registry) {
        if (fw.status == "CRITICAL_FIX") {
            std::cout << "  ⚠ " << comp << " v" << ver_str(fw)
                      << " [" << fw.hash << "]\n";
        }
    }

    // Print full registry
    std::cout << "\n[FIRMWARE REGISTRY]\n";
    std::cout << std::left
              << std::setw(15) << "Component"
              << std::setw(12) << "Version"
              << std::setw(12) << "Hash"
              << "Status\n";
    std::cout << std::string(55, '-') << "\n";
    for (auto& [comp, fw] : fw_registry) {
        std::cout << std::setw(15) << comp
                  << std::setw(12) << ("v" + ver_str(fw))
                  << std::setw(12) << fw.hash
                  << fw.status << "\n";
    }
}

// ─────────────────────────────────────────────────────────────
// 2. unordered_multimap: component → multiple changelog entries
// ─────────────────────────────────────────────────────────────
void demo_changelog() {
    std::cout << "\n=== [unordered_multimap] Firmware Changelog ===\n";

    std::unordered_multimap<std::string, std::string> changelog;

    // kernel changelog entries
    changelog.emplace("kernel", "v5.14: Added eBPF CO-RE support");
    changelog.emplace("kernel", "v5.15: EEVDF scheduler merged");
    changelog.emplace("kernel", "v5.16: KVM nested virt improvements");

    // gpu_drv changelog
    changelog.emplace("gpu_drv", "v1.2: DMA-BUF zero-copy added");
    changelog.emplace("gpu_drv", "v1.3: SMMU fault handling fixed");
    changelog.emplace("gpu_drv", "v1.4: Scheduler TDR watchdog added");

    // sensor_fw changelog
    changelog.emplace("sensor_fw", "v2.9: I2C threaded IRQ path");
    changelog.emplace("sensor_fw", "v3.0: SPI IMU driver merged");
    changelog.emplace("sensor_fw", "v3.0.1: ODR sysfs race fixed");

    // ota_manager
    changelog.emplace("ota_manager", "v1.1: TLS 1.2 ECDHE added");
    changelog.emplace("ota_manager", "v1.2: Anti-rollback counter fix");

    // Print changelog for a component
    auto print_log = [&](const std::string& comp) {
        auto [beg, fin] = changelog.equal_range(comp);
        std::cout << "[CHANGELOG: " << comp << "]\n";
        for (auto it = beg; it != fin; ++it) {
            std::cout << "  • " << it->second << "\n";
        }
        std::cout << "  (" << changelog.count(comp) << " entries)\n";
    };

    print_log("kernel");
    print_log("gpu_drv");
    print_log("ota_manager");

    // Add hotfix entry
    changelog.emplace("ota_manager", "v1.2.1: MITM cert check tightened");
    std::cout << "\n[HOTFIX] ota_manager entry added. "
              << "Total entries: " << changelog.count("ota_manager") << "\n";

    // Remove oldest kernel entry (obsolete)
    auto kr = changelog.equal_range("kernel");
    if (kr.first != kr.second) {
        std::cout << "[PRUNED] " << kr.first->second << "\n";
        changelog.erase(kr.first);
    }
}

// ──────────────────────────────────────────────────────────
// 3. unordered_set: verified firmware hashes (trust store)
// ──────────────────────────────────────────────────────────
void demo_hash_trust_store() {
    std::cout << "\n=== [unordered_set] Firmware Hash Trust Store ===\n";

    // Pre-loaded trusted hashes (would come from OTP/eFuse in real HW)
    std::unordered_set<std::string> trust_store = {
        "a1b2c3d4",  // bootloader v2.1.0
        "11223344",  // kernel v5.16.0
        "c9d0e1f2",  // gpu_drv v1.4.3
        "a3b4c5d6",  // sensor_fw v3.0.1
        "f7e8d9c0",  // ota_manager v1.2.0
        "aabbccdd",  // wifi_fw v4.0.0
    };

    // Simulate received firmware packages (some tampered)
    std::vector<std::pair<std::string, std::string>> packages = {
        {"kernel_v5.16.0.bin",       "11223344"},  // OK
        {"gpu_drv_v1.4.3.bin",       "c9d0e1f2"},  // OK
        {"evil_kernel.bin",           "deadbeef"},  // TAMPERED
        {"sensor_fw_v3.0.1.bin",     "a3b4c5d6"},  // OK
        {"fake_bootloader.bin",       "cafebabe"},  // TAMPERED
        {"ota_manager_v1.2.0.bin",   "f7e8d9c0"},  // OK
    };

    std::cout << std::left
              << std::setw(35) << "Package"
              << std::setw(12) << "Hash"
              << "Result\n";
    std::cout << std::string(60, '-') << "\n";

    int passed = 0, failed = 0;
    for (auto& [pkg, hash] : packages) {
        bool trusted = trust_store.count(hash) > 0;
        std::cout << std::setw(35) << pkg
                  << std::setw(12) << hash
                  << (trusted ? "✓ VERIFIED" : "✗ REJECTED") << "\n";
        trusted ? ++passed : ++failed;
    }

    std::cout << "\n[SUMMARY] Passed: " << passed
              << "  Failed: " << failed << "\n";

    // Revoke a compromised hash
    trust_store.erase("c9d0e1f2");
    std::cout << "[REVOKED] gpu_drv v1.4.3 hash c9d0e1f2 removed from trust store.\n";
    std::cout << "[TRUST STORE SIZE] " << trust_store.size() << "\n";
}

// ──────────────────────────────────────────────────────────
// 4. unordered_multiset: OTA update event frequency log
// ──────────────────────────────────────────────────────────
void demo_ota_event_log() {
    std::cout << "\n=== [unordered_multiset] OTA Update Event Log ===\n";

    std::unordered_multiset<std::string> event_log = {
        "DOWNLOAD_START",  "DOWNLOAD_OK",    "VERIFY_START",
        "VERIFY_OK",       "FLASH_START",     "FLASH_OK",
        "DOWNLOAD_START",  "DOWNLOAD_FAIL",   "RETRY",
        "DOWNLOAD_START",  "DOWNLOAD_OK",    "VERIFY_START",
        "VERIFY_FAIL",     "ROLLBACK",        "DOWNLOAD_START",
        "DOWNLOAD_OK",    "VERIFY_START",    "VERIFY_OK",
        "FLASH_START",    "FLASH_OK",        "BOOT_SUCCESS",
        "RETRY",           "RETRY"
    };

    // Print event summary
    std::unordered_set<std::string> unique(event_log.begin(),event_log.end());
    std::cout << std::left
              << std::setw(20) << "Event"
              << std::setw(8)  << "Count"
              << "Indicator\n";
    std::cout << std::string(45, '-') << "\n";
    for (auto& ev : unique) {
        size_t cnt = event_log.count(ev);
        std::string indicator = "";
        if (ev == "DOWNLOAD_FAIL") indicator = " ⚠ CHECK NETWORK";
        if (ev == "VERIFY_FAIL")   indicator = " ⚠ POSSIBLE TAMPER";
        if (ev == "ROLLBACK")      indicator = " ↩ ANTI-ROLLBACK";
        if (ev == "BOOT_SUCCESS")  indicator = " ✓ UPDATE COMPLETE";
        if (ev == "RETRY" && cnt > 2) indicator = " ⚠ EXCESSIVE RETRIES";
        std::cout << std::setw(20) << ev
                  << std::setw(8)  << cnt
                  << indicator << "\n";
    }

    // Success rate
    double starts   = event_log.count("DOWNLOAD_START");
    double successes = event_log.count("FLASH_OK");
    std::cout << "\n[OTA SUCCESS RATE] "
              << std::fixed << std::setprecision(0)
              << (successes / starts * 100.0) << "%"
              << " (" << (int)successes << "/" << (int)starts << " attempts)\n";
}

int main() {
    std::cout << "╔══════════════════════════════════════════╗\n";
    std::cout << "║    Firmware Update & Version Manager     ║\n";
    std::cout << "╚══════════════════════════════════════════╝\n";

    demo_firmware_map();
    demo_changelog();
    demo_hash_trust_store();
    demo_ota_event_log();

    return 0;
}
