#ifndef CAMERA_H
#define CAMERA_H

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include "Aircraft.h"

enum CameraMode {
    COCKPIT,        // First-person from cockpit
    CHASE,          // Third-person chase camera
    ORBIT,          // Free orbit around aircraft
    TOWER,          // Fixed position watching aircraft
    FREE            // Free flying camera
};

class Camera {
public:
    glm::vec3 position;
    glm::vec3 target;
    glm::vec3 up;
    
    CameraMode mode;
    
    // Chase camera parameters
    float chaseDistance;
    float chaseHeight;
    float chaseSmoothness;
    
    // Orbit camera parameters
    float orbitDistance;
    float orbitAngle;
    float orbitHeight;
    
    // Free camera parameters
    float yaw;
    float pitch;
    float freeSpeed;
    
    // Tower camera
    glm::vec3 towerPosition;
    
    Camera()
        : position(glm::vec3(0.0f, 5.0f, 10.0f)),
          target(glm::vec3(0.0f)),
          up(glm::vec3(0.0f, 1.0f, 0.0f)),
          mode(CHASE),
          chaseDistance(20.0f),
          chaseHeight(5.0f),
          chaseSmoothness(0.1f),
          orbitDistance(30.0f),
          orbitAngle(0.0f),
          orbitHeight(10.0f),
          yaw(-90.0f),
          pitch(0.0f),
          freeSpeed(20.0f),
          towerPosition(glm::vec3(0.0f, 50.0f, 200.0f))
    {
    }
    
    void Update(const Aircraft& aircraft, float deltaTime) {
        switch (mode) {
            case COCKPIT:
                UpdateCockpitView(aircraft);
                break;
            case CHASE:
                UpdateChaseView(aircraft, deltaTime);
                break;
            case ORBIT:
                UpdateOrbitView(aircraft, deltaTime);
                break;
            case TOWER:
                UpdateTowerView(aircraft);
                break;
            case FREE:
                // Free camera doesn't track aircraft
                break;
        }
    }
    
    glm::mat4 GetViewMatrix() const {
        return glm::lookAt(position, target, up);
    }
    
    void CycleMode() {
        mode = static_cast<CameraMode>((mode + 1) % 5);
    }
    
    void SetMode(CameraMode newMode) {
        mode = newMode;
    }
    
    // Free camera movement
    void MoveForward(float deltaTime) {
        glm::vec3 forward = glm::normalize(target - position);
        position += forward * freeSpeed * deltaTime;
        target += forward * freeSpeed * deltaTime;
    }
    
    void MoveBackward(float deltaTime) {
        glm::vec3 forward = glm::normalize(target - position);
        position -= forward * freeSpeed * deltaTime;
        target -= forward * freeSpeed * deltaTime;
    }
    
    void MoveLeft(float deltaTime) {
        glm::vec3 forward = glm::normalize(target - position);
        glm::vec3 right = glm::normalize(glm::cross(forward, up));
        position -= right * freeSpeed * deltaTime;
        target -= right * freeSpeed * deltaTime;
    }
    
    void MoveRight(float deltaTime) {
        glm::vec3 forward = glm::normalize(target - position);
        glm::vec3 right = glm::normalize(glm::cross(forward, up));
        position += right * freeSpeed * deltaTime;
        target += right * freeSpeed * deltaTime;
    }
    
    void MoveUp(float deltaTime) {
        position += up * freeSpeed * deltaTime;
        target += up * freeSpeed * deltaTime;
    }
    
    void MoveDown(float deltaTime) {
        position -= up * freeSpeed * deltaTime;
        target -= up * freeSpeed * deltaTime;
    }
    
    void ProcessMouseMovement(float xoffset, float yoffset) {
        if (mode == FREE) {
            const float sensitivity = 0.1f;
            yaw += xoffset * sensitivity;
            pitch += yoffset * sensitivity;
            
            // Constrain pitch
            if (pitch > 89.0f) pitch = 89.0f;
            if (pitch < -89.0f) pitch = -89.0f;
            
            UpdateFreeCamera();
        } else if (mode == ORBIT) {
            orbitAngle += xoffset * 0.5f;
            orbitHeight += yoffset * 0.1f;
            orbitHeight = glm::clamp(orbitHeight, -20.0f, 50.0f);
        }
    }
    
    void AdjustDistance(float delta) {
        if (mode == CHASE) {
            chaseDistance += delta;
            chaseDistance = glm::clamp(chaseDistance, 5.0f, 100.0f);
        } else if (mode == ORBIT) {
            orbitDistance += delta;
            orbitDistance = glm::clamp(orbitDistance, 10.0f, 200.0f);
        }
    }

private:
    void UpdateCockpitView(const Aircraft& aircraft) {
        // Position camera at cockpit location
        glm::vec3 cockpitOffset = glm::vec3(0.0f, 1.5f, -2.0f);
        position = aircraft.position + glm::rotate(aircraft.orientation, cockpitOffset);
        target = position + aircraft.GetForward() * 10.0f;
        up = aircraft.GetUp();
    }
    
    void UpdateChaseView(const Aircraft& aircraft, float deltaTime) {
        // Calculate ideal camera position behind aircraft
        glm::vec3 idealPosition = aircraft.position - 
                                  aircraft.GetForward() * chaseDistance +
                                  glm::vec3(0.0f, chaseHeight, 0.0f);
        
        // Smoothly interpolate to ideal position
        position = glm::mix(position, idealPosition, chaseSmoothness);
        
        // Look at aircraft
        target = aircraft.position;
        up = glm::vec3(0.0f, 1.0f, 0.0f);
    }
    
    void UpdateOrbitView(const Aircraft& aircraft, float deltaTime) {
        // Orbit around aircraft
        float angleRad = glm::radians(orbitAngle);
        position = aircraft.position + glm::vec3(
            std::cos(angleRad) * orbitDistance,
            orbitHeight,
            std::sin(angleRad) * orbitDistance
        );
        
        target = aircraft.position;
        up = glm::vec3(0.0f, 1.0f, 0.0f);
        
        // Auto-rotate
        orbitAngle += deltaTime * 10.0f;
    }
    
    void UpdateTowerView(const Aircraft& aircraft) {
        position = towerPosition;
        target = aircraft.position;
        up = glm::vec3(0.0f, 1.0f, 0.0f);
    }
    
    void UpdateFreeCamera() {
        glm::vec3 direction;
        direction.x = cos(glm::radians(yaw)) * cos(glm::radians(pitch));
        direction.y = sin(glm::radians(pitch));
        direction.z = sin(glm::radians(yaw)) * cos(glm::radians(pitch));
        
        glm::vec3 forward = glm::normalize(direction);
        target = position + forward * 10.0f;
    }
};

#endif