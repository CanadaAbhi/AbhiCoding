#ifndef AIRCRAFT_H
#define AIRCRAFT_H

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/quaternion.hpp>
#include <cmath>

class Aircraft {
public:
    // Position and orientation
    glm::vec3 position;
    glm::quat orientation;
    
    // Linear motion
    glm::vec3 velocity;
    glm::vec3 acceleration;
    
    // Angular motion
    glm::vec3 angularVelocity;  // pitch, yaw, roll rates (radians/sec)
    glm::vec3 angularAcceleration;
    
    // Flight parameters
    float throttle;         // 0.0 to 1.0
    float pitch;           // Elevator control (-1 to 1)
    float roll;            // Aileron control (-1 to 1)
    float yaw;             // Rudder control (-1 to 1)
    float flaps;           // 0.0 to 1.0
    float brake;           // 0.0 to 1.0
    
    // Aircraft properties
    float mass;            // kg
    float wingArea;        // m²
    float wingSpan;        // m
    float engineThrust;    // Newtons
    float dragCoefficient;
    float liftCoefficient;
    
    // Moments of inertia (simplified)
    float pitchInertia;
    float rollInertia;
    float yawInertia;
    
    // State
    bool onGround;
    float groundHeight;
    
    // Constants
    const float AIR_DENSITY = 1.225f;  // kg/m³ at sea level
    const float GRAVITY = 9.81f;       // m/s²
    
    Aircraft(glm::vec3 startPos = glm::vec3(0.0f, 100.0f, 0.0f))
        : position(startPos),
          orientation(glm::quat(1.0f, 0.0f, 0.0f, 0.0f)),
          velocity(glm::vec3(0.0f)),
          acceleration(glm::vec3(0.0f)),
          angularVelocity(glm::vec3(0.0f)),
          angularAcceleration(glm::vec3(0.0f)),
          throttle(0.0f),
          pitch(0.0f),
          roll(0.0f),
          yaw(0.0f),
          flaps(0.0f),
          brake(0.0f),
          mass(5000.0f),
          wingArea(30.0f),
          wingSpan(15.0f),
          engineThrust(50000.0f),
          dragCoefficient(0.025f),
          liftCoefficient(0.4f),
          pitchInertia(15000.0f),
          rollInertia(8000.0f),
          yawInertia(20000.0f),
          onGround(false),
          groundHeight(0.0f)
    {
    }
    
    // Update aircraft physics
    void Update(float deltaTime) {
        // Calculate forces in world space
        glm::vec3 totalForce = CalculateForces();
        
        // Calculate torques
        glm::vec3 totalTorque = CalculateTorques();
        
        // Update linear motion
        acceleration = totalForce / mass;
        velocity += acceleration * deltaTime;
        position += velocity * deltaTime;
        
        // Update angular motion
        angularAcceleration.x = totalTorque.x / pitchInertia;  // Pitch
        angularAcceleration.y = totalTorque.y / yawInertia;     // Yaw
        angularAcceleration.z = totalTorque.z / rollInertia;    // Roll
        
        angularVelocity += angularAcceleration * deltaTime;
        
        // Apply angular damping
        angularVelocity *= 0.98f;
        
        // Update orientation using quaternions
        glm::quat angularVelQuat(0.0f, 
                                 angularVelocity.x, 
                                 angularVelocity.y, 
                                 angularVelocity.z);
        glm::quat orientationDelta = 0.5f * angularVelQuat * orientation;
        orientation += orientationDelta * deltaTime;
        orientation = glm::normalize(orientation);
        
        // Ground collision
        if (position.y <= groundHeight + 2.0f) {
            position.y = groundHeight + 2.0f;
            
            if (velocity.y < 0.0f) {
                // Landing impact
                if (std::abs(velocity.y) > 3.0f) {
                    // Hard landing - bounce
                    velocity.y = -velocity.y * 0.3f;
                } else {
                    // Soft landing - stop vertical motion
                    velocity.y = 0.0f;
                    onGround = true;
                }
                
                // Apply ground friction
                if (onGround) {
                    velocity.x *= 0.95f;
                    velocity.z *= 0.95f;
                    angularVelocity *= 0.9f;
                }
            }
        } else {
            onGround = false;
        }
        
        // Clamp controls
        throttle = glm::clamp(throttle, 0.0f, 1.0f);
        pitch = glm::clamp(pitch, -1.0f, 1.0f);
        roll = glm::clamp(roll, -1.0f, 1.0f);
        yaw = glm::clamp(yaw, -1.0f, 1.0f);
        flaps = glm::clamp(flaps, 0.0f, 1.0f);
        brake = glm::clamp(brake, 0.0f, 1.0f);
    }
    
    // Get forward vector
    glm::vec3 GetForward() const {
        return glm::rotate(orientation, glm::vec3(0.0f, 0.0f, -1.0f));
    }
    
    // Get right vector
    glm::vec3 GetRight() const {
        return glm::rotate(orientation, glm::vec3(1.0f, 0.0f, 0.0f));
    }
    
    // Get up vector
    glm::vec3 GetUp() const {
        return glm::rotate(orientation, glm::vec3(0.0f, 1.0f, 0.0f));
    }
    
    // Get Euler angles (pitch, yaw, roll)
    glm::vec3 GetEulerAngles() const {
        return glm::eulerAngles(orientation);
    }
    
    // Get airspeed
    float GetAirspeed() const {
        return glm::length(velocity);
    }
    
    // Get altitude above ground
    float GetAltitude() const {
        return position.y - groundHeight;
    }
    
    // Get view matrix for camera
    glm::mat4 GetViewMatrix() const {
        glm::vec3 center = position + GetForward();
        return glm::lookAt(position, center, GetUp());
    }
    
    // Get model matrix for rendering aircraft
    glm::mat4 GetModelMatrix() const {
        glm::mat4 model = glm::mat4(1.0f);
        model = glm::translate(model, position);
        model *= glm::mat4_cast(orientation);
        return model;
    }

private:
    glm::vec3 CalculateForces() {
        glm::vec3 totalForce(0.0f);
        
        // Gravity (always downward in world space)
        glm::vec3 gravityForce = glm::vec3(0.0f, -mass * GRAVITY, 0.0f);
        totalForce += gravityForce;
        
        // Thrust (along aircraft's forward direction)
        glm::vec3 thrustForce = GetForward() * throttle * engineThrust;
        totalForce += thrustForce;
        
        float airspeed = GetAirspeed();
        
        if (airspeed > 0.1f) {
            // Dynamic pressure
            float dynamicPressure = 0.5f * AIR_DENSITY * airspeed * airspeed;
            
            // Lift (perpendicular to velocity, in the up direction of aircraft)
            float angleOfAttack = CalculateAngleOfAttack();
            float liftCoeff = liftCoefficient * (1.0f + flaps * 0.5f) * 
                             std::sin(angleOfAttack * 2.0f);
            float liftMagnitude = liftCoeff * dynamicPressure * wingArea;
            
            // Lift is perpendicular to velocity
            glm::vec3 liftDirection = glm::normalize(glm::cross(
                glm::cross(GetUp(), velocity),
                velocity
            ));
            glm::vec3 liftForce = liftDirection * liftMagnitude;
            totalForce += liftForce;
            
            // Drag (opposite to velocity)
            float dragCoeff = dragCoefficient + 
                             std::abs(angleOfAttack) * 0.1f +
                             flaps * 0.05f +
                             brake * 0.3f;
            float dragMagnitude = dragCoeff * dynamicPressure * wingArea;
            glm::vec3 dragForce = -glm::normalize(velocity) * dragMagnitude;
            totalForce += dragForce;
        }
        
        // Ground effect (increased lift near ground)
        if (GetAltitude() < wingSpan && !onGround) {
            float groundEffect = (1.0f - GetAltitude() / wingSpan) * 0.3f;
            totalForce.y += groundEffect * mass * GRAVITY;
        }
        
        return totalForce;
    }
    
    glm::vec3 CalculateTorques() {
        glm::vec3 totalTorque(0.0f);
        
        float airspeed = GetAirspeed();
        
        if (airspeed > 5.0f) {
            float controlAuthority = glm::min(airspeed / 50.0f, 1.0f);
            
            // Pitch torque (elevator)
            float pitchTorque = pitch * controlAuthority * 50000.0f;
            totalTorque.x = pitchTorque;
            
            // Yaw torque (rudder)
            float yawTorque = yaw * controlAuthority * 30000.0f;
            totalTorque.y = yawTorque;
            
            // Roll torque (ailerons)
            float rollTorque = roll * controlAuthority * 40000.0f;
            totalTorque.z = rollTorque;
            
            // Stability augmentation (prevents over-rotation)
            totalTorque.x -= angularVelocity.x * 5000.0f;
            totalTorque.y -= angularVelocity.y * 3000.0f;
            totalTorque.z -= angularVelocity.z * 4000.0f;
        }
        
        return totalTorque;
    }
    
    float CalculateAngleOfAttack() {
        if (glm::length(velocity) < 0.1f) return 0.0f;
        
        glm::vec3 forward = GetForward();
        glm::vec3 velNorm = glm::normalize(velocity);
        
        // Angle between forward and velocity
        float dot = glm::dot(forward, velNorm);
        dot = glm::clamp(dot, -1.0f, 1.0f);
        
        // Calculate signed angle
        glm::vec3 right = GetRight();
        float cross = glm::dot(glm::cross(forward, velNorm), right);
        
        float angle = std::acos(dot);
        if (cross < 0.0f) angle = -angle;
        
        return angle;
    }
};

#endif