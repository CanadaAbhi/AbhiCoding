#ifndef HUD_H
#define HUD_H

#include <string>
#include <sstream>
#include <iomanip>
#include "Aircraft.h"

class HUD {
public:
    static std::string GetSpeedString(const Aircraft& aircraft) {
        float speed = aircraft.GetAirspeed() * 1.94384f; // Convert m/s to knots
        std::ostringstream oss;
        oss << "SPEED: " << std::fixed << std::setprecision(0) << speed << " kts";
        return oss.str();
    }
    
    static std::string GetAltitudeString(const Aircraft& aircraft) {
        float altitude = aircraft.GetAltitude() * 3.28084f; // Convert m to feet
        std::ostringstream oss;
        oss << "ALT: " << std::fixed << std::setprecision(0) << altitude << " ft";
        return oss.str();
    }
    
    static std::string GetThrottleString(const Aircraft& aircraft) {
        std::ostringstream oss;
        oss << "THROTTLE: " << std::fixed << std::setprecision(0) 
            << (aircraft.throttle * 100.0f) << "%";
        return oss.str();
    }
    
    static std::string GetAttitudeString(const Aircraft& aircraft) {
        glm::vec3 euler = aircraft.GetEulerAngles();
        float pitch = glm::degrees(euler.x);
        float roll = glm::degrees(euler.z);
        
        std::ostringstream oss;
        oss << "PITCH: " << std::fixed << std::setprecision(1) << pitch << "° "
            << "ROLL: " << std::fixed << std::setprecision(1) << roll << "°";
        return oss.str();
    }
    
    static std::string GetHeadingString(const Aircraft& aircraft) {
        glm::vec3 euler = aircraft.GetEulerAngles();
        float heading = glm::degrees(euler.y);
        while (heading < 0) heading += 360.0f;
        while (heading >= 360) heading -= 360.0f;
        
        std::ostringstream oss;
        oss << "HDG: " << std::fixed << std::setprecision(0) << heading << "°";
        return oss.str();
    }
    
    static std::string GetVerticalSpeedString(const Aircraft& aircraft) {
        float vs = aircraft.velocity.y * 196.85f; // Convert m/s to ft/min
        std::ostringstream oss;
        oss << "VS: " << std::fixed << std::setprecision(0) << vs << " fpm";
        return oss.str();
    }
    
    static std::string GetGForceString(const Aircraft& aircraft) {
        float gForce = glm::length(aircraft.acceleration) / 9.81f;
        std::ostringstream oss;
        oss << "G: " << std::fixed << std::setprecision(2) << gForce;
        return oss.str();
    }
    
    static std::string GetFlapsString(const Aircraft& aircraft) {
        std::ostringstream oss;
        oss << "FLAPS: " << std::fixed << std::setprecision(0) 
            << (aircraft.flaps * 100.0f) << "%";
        return oss.str();
    }
    
    static std::string GetGroundStatusString(const Aircraft& aircraft) {
        return aircraft.onGround ? "ON GROUND" : "AIRBORNE";
    }
    
    static std::string GetAngleOfAttackString(const Aircraft& aircraft) {
        // Calculate AoA
        if (glm::length(aircraft.velocity) < 0.1f) return "AOA: N/A";
        
        glm::vec3 forward = aircraft.GetForward();
        glm::vec3 velNorm = glm::normalize(aircraft.velocity);
        
        float dot = glm::dot(forward, velNorm);
        dot = glm::clamp(dot, -1.0f, 1.0f);
        
        float angle = glm::degrees(std::acos(dot));
        
        std::ostringstream oss;
        oss << "AOA: " << std::fixed << std::setprecision(1) << angle << "°";
        return oss.str();
    }
    
    // Simple artificial horizon representation (text-based)
    static void GetArtificialHorizon(const Aircraft& aircraft, 
                                     std::vector<std::string>& lines) {
        lines.clear();
        glm::vec3 euler = aircraft.GetEulerAngles();
        float pitch = glm::degrees(euler.x);
        float roll = glm::degrees(euler.z);
        
        // Simplified horizon display
        int centerLine = 5;
        int pitchOffset = static_cast<int>(pitch / 10.0f);
        
        for (int i = 0; i < 11; i++) {
            std::string line = "";
            int lineAngle = (centerLine - i + pitchOffset) * 10;
            
            if (i == centerLine) {
                line = "======[O]======"; // Aircraft reference
            } else if (lineAngle == 0) {
                line = "---------------"; // Horizon
            } else if (lineAngle > 0) {
                line = "- - - - - - - -"; // Sky
            } else {
                line = "_______________"; // Ground
            }
            
            lines.push_back(line);
        }
    }
};

#endif