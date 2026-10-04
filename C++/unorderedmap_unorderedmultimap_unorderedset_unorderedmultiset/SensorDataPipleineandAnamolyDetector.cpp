#include <iostream>
#include <unordered_map>
#include <unordered_multimap>
#include <unordered_set>
#include <unordered_multiset>
#include <string>
#include <vector>
#include <numeric>
#include <iomanip>
#include <cmath>
#include <algorithm>

// ──────────────────────────────────────────────────────
// 1. unordered_map: sensor_id → sensor configuration
// ──────────────────────────────────────────────────────
struct SensorConfig {
    std::string type;        // TEMP, IMU, PRESSURE, HUMIDITY
    std::string protocol;    // I2C, SPI, ADC
    int         odr_hz;
    double      threshold;   // anomaly threshold
    bool        enabled;
};

void demo_sensor_config() {
    std::cout << "\n=== [unordered_map] Sensor Configuration Registry ===\n";

    std::unordered_map<std::string, SensorConfig> sensor_cfg = {
        {"temp_0",   {"TEMP",     "I2C", 10,  85.0,  true}},
        {"temp_1",   {"TEMP",     "I2C", 10,  85.0,  true}},
        {"imu_0",    {"IMU",      "SPI", 200, 16.0,  true}},
        {"pres_0",   {"PRESSURE", "I2C", 50,  110.0, true}},
        {"humid_0",  {"HUMIDITY", "ADC", 5,   95.0,  false}},
    };

    // Enable a disabled sensor
    sensor_cfg["humid_0"].enabled = true;
    std::cout << "[ENABLED] humid_0\n";

    // Change ODR for IMU (power saving mode)
    sensor_cfg["imu_0"].odr_hz = 50;
    std::cout << "[ODR CHANGE] imu_0 → 50 Hz (power save)\n";

    // Add a new sensor at runtime (hot-plug)
    sensor_cfg["temp_2"] = {"TEMP", "SPI", 20, 80.0, true};
    std::cout << "[HOT-PLUG] temp_2 registered\n";

    // Print configuration table
    std::cout << "\n" << std::left
              << std::setw(12) << "Sensor"
              << std::setw(12) << "Type"
              << std::setw(8)  << "Proto"
              << std::setw(8)  << "ODR(Hz)"
              << std::setw(12) << "Threshold"
              << "Status\n";
    std::cout << std::string(60, '-') << "\n";
    for (auto& [id, cfg] : sensor_cfg) {
        std::cout << std::setw(12) << id
                  << std::setw(12) << cfg.type
                  << std::setw(8)  << cfg.protocol
                  << std::setw(8)  << cfg.odr_hz
                  << std::setw(12) << cfg.threshold
                  << (cfg.enabled ? "ENABLED" : "DISABLED") << "\n";
    }
}

// ─────────────────────────────────────────────────────────
// 2. unordered_multimap: sensor_id → multiple readings
// ─────────────────────────────────────────────────────────
void demo_sensor_readings() {
    std::cout << "\n=== [unordered_multimap] Sensor Reading Buffer ===\n";

    // Key = sensor_id, Value = reading value as string "value@timestamp"
    std::unordered_multimap<std::string, double> readings;

    // Batch of readings arriving
    std::vector<std::pair<std::string, double>> batch = {
        {"temp_0", 72.3}, {"temp_0", 73.1}, {"temp_0", 74.5},
        {"temp_0", 86.2}, {"temp_0", 85.9},  // above threshold
        {"imu_0",  9.81}, {"imu_0", 10.2},  {"imu_0", 9.77},
        {"pres_0", 101.3},{"pres_0", 102.1},{"pres_0", 111.5}, // above
        {"temp_1", 45.0}, {"temp_1", 44.8}, {"temp_1", 45.3},
    };

    for (auto& [id, val] : batch) readings.emplace(id, val);

    // Compute stats for each sensor
    std::unordered_set<std::string> unique_ids;
    for (auto& [id, _] : readings) unique_ids.insert(id);

    std::cout << std::left
              << std::setw(12) << "Sensor"
              << std::setw(10) << "Samples"
              << std::setw(10) << "Min"
              << std::setw(10) << "Max"
              << std::setw(10) << "Avg"
              << "Status\n";
    std::cout << std::string(60, '-') << "\n";

    // Thresholds per sensor
    std::unordered_map<std::string, double> thresholds = {
        {"temp_0", 85.0}, {"temp_1", 85.0},
        {"imu_0",  16.0}, {"pres_0", 110.0}
    };

    for (auto& sid : unique_ids) {
        auto [beg, fin] = readings.equal_range(sid);
        std::vector<double> vals;
        for (auto it = beg; it != fin; ++it) vals.push_back(it->second);

        double mn  = *std::min_element(vals.begin(), vals.end());
        double mx  = *std::max_element(vals.begin(), vals.end());
        double avg = std::accumulate(vals.begin(), vals.end(), 0.0) / vals.size();
        double thr = thresholds.count(sid) ? thresholds[sid] : 999.0;
        std::string st = (mx > thr) ? "⚠ ANOMALY" : "OK";

        std::cout << std::setw(12) << sid
                  << std::setw(10) << vals.size()
                  << std::setw(10) << std::fixed << std::setprecision(1) << mn
                  << std::setw(10) << mx
                  << std::setw(10) << avg
                  << st << "\n";
    }
}

// ──────────────────────────────────────────────────────────
// 3. unordered_set: anomalous sensor IDs (quarantine list)
// ──────────────────────────────────────────────────────────
void demo_anomaly_quarantine() {
    std::cout << "\n=== [unordered_set] Sensor Anomaly Quarantine ===\n";

    std::unordered_set<std::string> quarantine;
    std::unordered_set<std::string> cleared;

    // Simulated anomaly detection results
    std::vector<std::pair<std::string, bool>> detections = {
        {"temp_0",  true},   // anomaly
        {"temp_1",  false},
        {"imu_0",   false},
        {"pres_0",  true},   // anomaly
        {"humid_0", true},   // anomaly
        {"temp_2",  false},
        {"temp_0",  true},   // duplicate alarm
    };

    std::cout << "[ANOMALY DETECTION LOG]\n";
    for (auto& [sid, is_anomaly] : detections) {
        if (is_anomaly) {
            auto [it, inserted] = quarantine.insert(sid);
            if (inserted) {
                std::cout << "  [QUARANTINE] " << sid << " isolated\n";
            } else {
                std::cout << "  [REPEAT]     " << sid << " still anomalous\n";
            }
        } else {
            if (quarantine.count(sid)) {
                std::cout << "  [CLEARED]    " << sid << " restored\n";
                quarantine.erase(sid);
                cleared.insert(sid);
            } else {
                std::cout << "  [OK]         " << sid << "\n";
            }
        }
    }

    std::cout << "\n[QUARANTINE LIST] " << quarantine.size() << " sensor(s):\n";
    for (auto& sid : quarantine) std::cout << "  • " << sid << "\n";

    std::cout << "\n[CLEARED LIST] " << cleared.size() << " sensor(s):\n";
    for (auto& sid : cleared) std::cout << "  • " << sid << "\n";
}

// ──────────────────────────────────────────────────────────
// 4. unordered_multiset: alert type frequency & prioritization
// ──────────────────────────────────────────────────────────
void demo_alert_frequency() {
    std::cout << "\n=== [unordered_multiset] Alert Type Frequency ===\n";

    std::unordered_multiset<std::string> alerts = {
        "TEMP_HIGH",     "TEMP_HIGH",    "PRESS_HIGH",
        "TEMP_HIGH",     "IMU_SHOCK",    "TEMP_HIGH",
        "HUMID_WARN",    "PRESS_HIGH",   "TEMP_HIGH",
        "SENSOR_FAULT",  "IMU_SHOCK",    "TEMP_HIGH",
        "PRESS_HIGH",    "TEMP_HIGH",    "HUMID_WARN",
        "SENSOR_FAULT",  "IMU_SHOCK",    "TEMP_HIGH",
        "TEMP_HIGH",     "PRESS_HIGH"
    };

    std::unordered_set<std::string> unique(alerts.begin(), alerts.end());

    // Sort by frequency for display
    std::vector<std::pair<std::string, size_t>> freq_vec;
    for (auto& a : unique) freq_vec.push_back({a, alerts.count(a)});
    std::sort(freq_vec.begin(), freq_vec.end(),
              [](auto& a, auto& b){ return a.second > b.second; });

    std::cout << std::left
              << std::setw(18) << "Alert Type"
              << std::setw(8)  << "Count"
              << std::setw(10) << "Priority"
              << "Bar\n";
    std::cout << std::string(55, '-') << "\n";

    for (auto& [alert, cnt] : freq_vec) {
        std::string priority =
            (alert == "SENSOR_FAULT") ? "P0-CRITICAL"  :
            (alert == "TEMP_HIGH"   ) ? "P1-HIGH"      :
            (alert == "PRESS_HIGH"  ) ? "P1-HIGH"      :
            (alert == "IMU_SHOCK"   ) ? "P2-MEDIUM"    :
                                        "P3-LOW";
        std::string bar(cnt, '█');
        std::cout << std::setw(18) << alert
                  << std::setw(8)  << cnt
                  << std::setw(10) << priority
                  << bar << "\n";
    }

    // Suppress all HUMID_WARN (below threshold reconfigured)
    size_t suppressed = alerts.count("HUMID_WARN");
    alerts.erase("HUMID_WARN");
    std::cout << "\n[SUPPRESSED] " << suppressed
              << " HUMID_WARN alerts (threshold updated)\n";

    // Check if any SENSOR_FAULT remains → escalate
    if (alerts.count("SENSOR_FAULT") > 0) {
        std::cout << "[ESCALATE] " << alerts.count("SENSOR_FAULT")
                  << " SENSOR_FAULT(s) → paging on-call engineer\n";
    }

    std::cout << "[ALERT QUEUE] " << alerts.size() << " alerts pending\n";
}

int main() {
    std::cout << "╔══════════════════════════════════════════╗\n";
    std::cout << "║    Sensor Pipeline & Anomaly Detector    ║\n";
    std::cout << "╚══════════════════════════════════════════╝\n";

    demo_sensor_config();
    demo_sensor_readings();
    demo_anomaly_quarantine();
    demo_alert_frequency();

    return 0;
}
