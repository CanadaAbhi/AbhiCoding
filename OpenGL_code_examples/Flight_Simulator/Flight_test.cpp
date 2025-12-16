#include "Aircraft.h"
#include "HUD.h"
#include <iostream>
#include <iomanip>
#include <thread>
#include <chrono>

void clearScreen() {
#ifdef _WIN32
    system("cls");
#else
    system("clear");
#endif
}

void printAircraftStatus(const Aircraft& aircraft, float elapsedTime) {
    clearScreen();
    
    std::cout << "╔═══════════════════════════════════════════════════════════╗\n";
    std::cout << "║           FLIGHT SIMULATOR - PHYSICS TEST                ║\n";
    std::cout << "╚═══════════════════════════════════════════════════════════╝\n\n";
    
    std::cout << "Flight Time: " << std::fixed << std::setprecision(1) 
              << elapsedTime << " seconds\n\n";
    
    // Primary flight instruments
    std::cout << "┌─ PRIMARY INSTRUMENTS ─────────────────────────────────────┐\n";
    std::cout << "│ " << std::setw(30) << std::left << HUD::GetSpeedString(aircraft) 
              << std::setw(30) << HUD::GetAltitudeString(aircraft) << "│\n";
    std::cout << "│ " << std::setw(30) << HUD::GetHeadingString(aircraft)
              << std::setw(30) << HUD::GetVerticalSpeedString(aircraft) << "│\n";
    std::cout << "└───────────────────────────────────────────────────────────┘\n\n";
    
    // Attitude
    std::cout << "┌─ ATTITUDE ────────────────────────────────────────────────┐\n";
    std::cout << "│ " << HUD::GetAttitudeString(aircraft) << std::string(20, ' ') << "│\n";
    std::cout << "│ " << HUD::GetAngleOfAttackString(aircraft) << std::string(35, ' ') << "│\n";
    std::cout << "└───────────────────────────────────────────────────────────┘\n\n";
    
    // Engine & systems
    std::cout << "┌─ ENGINE & SYSTEMS ────────────────────────────────────────┐\n";
    std::cout << "│ " << std::setw(30) << HUD::GetThrottleString(aircraft)
              << std::setw(30) << HUD::GetFlapsString(aircraft) << "│\n";
    std::cout << "│ " << std::setw(30) << HUD::GetGForceString(aircraft)
              << std::setw(30) << HUD::GetGroundStatusString(aircraft) << "│\n";
    std::cout << "└───────────────────────────────────────────────────────────┘\n\n";
    
    // Position
    std::cout << "┌─ POSITION ────────────────────────────────────────────────┐\n";
    std::cout << "│ X: " << std::setw(10) << std::fixed << std::setprecision(2) 
              << aircraft.position.x << " m    ";
    std::cout << "Y: " << std::setw(10) << aircraft.position.y << " m    ";
    std::cout << "Z: " << std::setw(10) << aircraft.position.z << " m  │\n";
    std::cout << "└───────────────────────────────────────────────────────────┘\n\n";
    
    // Velocity
    std::cout << "┌─ VELOCITY ────────────────────────────────────────────────┐\n";
    std::cout << "│ VX: " << std::setw(9) << aircraft.velocity.x << " m/s  ";
    std::cout << "VY: " << std::setw(9) << aircraft.velocity.y << " m/s  ";
    std::cout << "VZ: " << std::setw(9) << aircraft.velocity.z << " m/s │\n";
    std::cout << "└───────────────────────────────────────────────────────────┘\n\n";
    
    // Controls
    std::cout << "┌─ CONTROL INPUTS ──────────────────────────────────────────┐\n";
    std::cout << "│ Pitch: " << std::setw(6) << std::setprecision(2) << aircraft.pitch 
              << "  Roll: " << std::setw(6) << aircraft.roll
              << "  Yaw: " << std::setw(6) << aircraft.yaw << std::string(15, ' ') << "│\n";
    std::cout << "└───────────────────────────────────────────────────────────┘\n\n";
}

void runSimulation() {
    Aircraft aircraft(glm::vec3(0.0f, 1000.0f, 0.0f)); // Start at 1000m altitude
    aircraft.groundHeight = 0.0f;
    
    std::cout << "╔═══════════════════════════════════════════════════════════╗\n";
    std::cout << "║        FLIGHT SIMULATOR - PHYSICS DEMONSTRATION           ║\n";
    std::cout << "╚═══════════════════════════════════════════════════════════╝\n\n";
    std::cout << "This demonstration will show basic flight physics:\n\n";
    std::cout << "Test 1: Gravity and free fall\n";
    std::cout << "Test 2: Throttle and climb\n";
    std::cout << "Test 3: Pitch control\n";
    std::cout << "Test 4: Roll maneuver\n";
    std::cout << "Test 5: Landing sequence\n\n";
    std::cout << "Press Enter to start...\n";
    std::cin.get();
    
    float deltaTime = 0.016f; // ~60 FPS
    float elapsedTime = 0.0f;
    
    // Test 1: Gravity (5 seconds)
    std::cout << "\n=== TEST 1: GRAVITY & FREE FALL ===\n";
    std::cout << "Aircraft starting at 1000m, no thrust...\n";
    std::this_thread::sleep_for(std::chrono::seconds(2));
    
    for (int i = 0; i < 300; i++) {
        aircraft.Update(deltaTime);
        elapsedTime += deltaTime;
        
        if (i % 30 == 0) {
            printAircraftStatus(aircraft, elapsedTime);
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
    }
    
    // Test 2: Throttle and climb
    std::cout << "\n=== TEST 2: THROTTLE & CLIMB ===\n";
    std::cout << "Increasing throttle to 80%...\n";
    std::this_thread::sleep_for(std::chrono::seconds(2));
    
    aircraft.throttle = 0.8f;
    for (int i = 0; i < 300; i++) {
        aircraft.Update(deltaTime);
        elapsedTime += deltaTime;
        
        if (i % 30 == 0) {
            printAircraftStatus(aircraft, elapsedTime);
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
    }
    
    // Test 3: Pitch control
    std::cout << "\n=== TEST 3: PITCH CONTROL ===\n";
    std::cout << "Pulling up (positive pitch)...\n";
    std::this_thread::sleep_for(std::chrono::seconds(2));
    
    aircraft.pitch = 0.5f;
    for (int i = 0; i < 300; i++) {
        aircraft.Update(deltaTime);
        elapsedTime += deltaTime;
        
        if (i % 30 == 0) {
            printAircraftStatus(aircraft, elapsedTime);
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
    }
    
    // Test 4: Roll maneuver
    std::cout << "\n=== TEST 4: ROLL MANEUVER ===\n";
    std::cout << "Banking right...\n";
    std::this_thread::sleep_for(std::chrono::seconds(2));
    
    aircraft.pitch = 0.0f;
    aircraft.roll = 0.7f;
    for (int i = 0; i < 300; i++) {
        aircraft.Update(deltaTime);
        elapsedTime += deltaTime;
        
        if (i % 30 == 0) {
            printAircraftStatus(aircraft, elapsedTime);
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
    }
    
    // Test 5: Landing sequence
    std::cout << "\n=== TEST 5: LANDING SEQUENCE ===\n";
    std::cout << "Descending for landing...\n";
    std::this_thread::sleep_for(std::chrono::seconds(2));
    
    aircraft.roll = 0.0f;
    aircraft.pitch = -0.2f;
    aircraft.throttle = 0.3f;
    aircraft.flaps = 1.0f;
    
    while (aircraft.GetAltitude() > 1.0f) {
        aircraft.Update(deltaTime);
        elapsedTime += deltaTime;
        
        static int frame = 0;
        if (frame++ % 30 == 0) {
            printAircraftStatus(aircraft, elapsedTime);
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
        
        // Flare before touchdown
        if (aircraft.GetAltitude() < 20.0f) {
            aircraft.pitch = 0.1f;
            aircraft.throttle = 0.1f;
        }
    }
    
    aircraft.brake = 1.0f;
    for (int i = 0; i < 100; i++) {
        aircraft.Update(deltaTime);
        elapsedTime += deltaTime;
        
        if (i % 30 == 0) {
            printAircraftStatus(aircraft, elapsedTime);
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
    }
    
    std::cout << "\n╔═══════════════════════════════════════════════════════════╗\n";
    std::cout << "║                 DEMONSTRATION COMPLETE                    ║\n";
    std::cout << "╚═══════════════════════════════════════════════════════════╝\n\n";
    std::cout << "Total flight time: " << std::fixed << std::setprecision(1) 
              << elapsedTime << " seconds\n";
    std::cout << "Landing speed: " << (aircraft.GetAirspeed() * 1.94384f) << " knots\n";
    std::cout << "Final altitude: " << (aircraft.GetAltitude() * 3.28084f) << " feet\n\n";
}

void printPhysicsInfo() {
    std::cout << "\n╔═══════════════════════════════════════════════════════════╗\n";
    std::cout << "║          FLIGHT PHYSICS IMPLEMENTATION DETAILS            ║\n";
    std::cout << "╚═══════════════════════════════════════════════════════════╝\n\n";
    
    std::cout << "FORCES SIMULATED:\n";
    std::cout << "  • Gravity: F = -9.81 * mass (downward)\n";
    std::cout << "  • Thrust: F = throttle * engineThrust (forward)\n";
    std::cout << "  • Lift: F = 0.5 * ρ * v² * S * CL (perpendicular to velocity)\n";
    std::cout << "  • Drag: F = 0.5 * ρ * v² * S * CD (opposite to velocity)\n\n";
    
    std::cout << "TORQUES SIMULATED:\n";
    std::cout << "  • Pitch torque (elevator): Controlled by W/S keys\n";
    std::cout << "  • Roll torque (ailerons): Controlled by A/D keys\n";
    std::cout << "  • Yaw torque (rudder): Controlled by Q/E keys\n\n";
    
    std::cout << "PHYSICS PARAMETERS:\n";
    std::cout << "  • Air density: 1.225 kg/m³\n";
    std::cout << "  • Mass: 5000 kg\n";
    std::cout << "  • Wing area: 30 m²\n";
    std::cout << "  • Max thrust: 50,000 N\n";
    std::cout << "  • Lift coefficient: 0.4\n";
    std::cout << "  • Drag coefficient: 0.025\n\n";
    
    std::cout << "ADDITIONAL EFFECTS:\n";
    std::cout << "  • Ground effect (increased lift near ground)\n";
    std::cout << "  • Angle of attack influences\n";
    std::cout << "  • Flaps increase lift\n";
    std::cout << "  • Brakes increase drag\n";
    std::cout << "  • Stability augmentation (damping)\n\n";
}

int main() {
    int choice;
    
    while (true) {
        clearScreen();
        std::cout << "╔═══════════════════════════════════════════════════════════╗\n";
        std::cout << "║              FLIGHT SIMULATOR - PHYSICS TEST              ║\n";
        std::cout << "╚═══════════════════════════════════════════════════════════╝\n\n";
        std::cout << "1. Run flight demonstration\n";
        std::cout << "2. View physics implementation details\n";
        std::cout << "3. Manual control test (coming soon)\n";
        std::cout << "4. Exit\n\n";
        std::cout << "Choose an option: ";
        
        std::cin >> choice;
        std::cin.ignore();
        
        switch (choice) {
            case 1:
                runSimulation();
                std::cout << "\nPress Enter to continue...";
                std::cin.get();
                break;
            case 2:
                printPhysicsInfo();
                std::cout << "\nPress Enter to continue...";
                std::cin.get();
                break;
            case 3:
                std::cout << "\nManual control requires the full OpenGL version.\n";
                std::cout << "Build and run flight_sim_main.cpp for interactive flight.\n\n";
                std::cout << "Press Enter to continue...";
                std::cin.get();
                break;
            case 4:
                std::cout << "\nThank you for testing the flight simulator!\n";
                return 0;
            default:
                std::cout << "\nInvalid choice. Press Enter to continue...";
                std::cin.get();
        }
    }
    
    return 0;
}