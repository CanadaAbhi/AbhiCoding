#ifndef COLLISION_H
#define COLLISION_H

#include <glm/glm.hpp>
#include <vector>

// Structure for AABB (Axis-Aligned Bounding Box)
struct AABB {
    glm::vec3 min;
    glm::vec3 max;
    
    AABB(glm::vec3 minimum, glm::vec3 maximum) : min(minimum), max(maximum) {}
};

// Structure for Sphere collider
struct Sphere {
    glm::vec3 center;
    float radius;
    
    Sphere(glm::vec3 c, float r) : center(c), radius(r) {}
};

// Plane for floor/wall collision
struct Plane {
    glm::vec3 normal;
    float distance;
    
    Plane(glm::vec3 n, float d) : normal(glm::normalize(n)), distance(d) {}
};

class CollisionManager {
public:
    std::vector<AABB> boxes;
    std::vector<Sphere> spheres;
    std::vector<Plane> planes;
    
    // Add collision objects
    void AddBox(glm::vec3 min, glm::vec3 max) {
        boxes.push_back(AABB(min, max));
    }
    
    void AddSphere(glm::vec3 center, float radius) {
        spheres.push_back(Sphere(center, radius));
    }
    
    void AddPlane(glm::vec3 normal, float distance) {
        planes.push_back(Plane(normal, distance));
    }
    
    // Check if point is colliding with any object
    bool CheckCollision(glm::vec3 position, float radius, glm::vec3& collisionNormal) {
        // Check box collisions
        for (const auto& box : boxes) {
            if (CheckPointAABB(position, radius, box, collisionNormal)) {
                return true;
            }
        }
        
        // Check sphere collisions
        for (const auto& sphere : spheres) {
            if (CheckPointSphere(position, radius, sphere, collisionNormal)) {
                return true;
            }
        }
        
        // Check plane collisions
        for (const auto& plane : planes) {
            if (CheckPointPlane(position, radius, plane, collisionNormal)) {
                return true;
            }
        }
        
        return false;
    }
    
private:
    bool CheckPointAABB(glm::vec3 point, float radius, const AABB& box, glm::vec3& normal) {
        // Find closest point on box to the sphere center
        glm::vec3 closestPoint = glm::clamp(point, box.min, box.max);
        
        // Calculate distance
        float distance = glm::length(point - closestPoint);
        
        if (distance < radius) {
            // Calculate collision normal
            glm::vec3 diff = point - closestPoint;
            if (glm::length(diff) > 0.0001f) {
                normal = glm::normalize(diff);
            } else {
                // Point is inside box, find closest face
                glm::vec3 distToMin = point - box.min;
                glm::vec3 distToMax = box.max - point;
                
                float minDist = glm::min(
                    glm::min(distToMin.x, distToMin.y),
                    glm::min(distToMin.z, glm::min(distToMax.x, glm::min(distToMax.y, distToMax.z)))
                );
                
                if (minDist == distToMin.x) normal = glm::vec3(-1, 0, 0);
                else if (minDist == distToMax.x) normal = glm::vec3(1, 0, 0);
                else if (minDist == distToMin.y) normal = glm::vec3(0, -1, 0);
                else if (minDist == distToMax.y) normal = glm::vec3(0, 1, 0);
                else if (minDist == distToMin.z) normal = glm::vec3(0, 0, -1);
                else normal = glm::vec3(0, 0, 1);
            }
            return true;
        }
        
        return false;
    }
    
    bool CheckPointSphere(glm::vec3 point, float radius, const Sphere& sphere, glm::vec3& normal) {
        float distance = glm::length(point - sphere.center);
        
        if (distance < radius + sphere.radius) {
            normal = glm::normalize(point - sphere.center);
            return true;
        }
        
        return false;
    }
    
    bool CheckPointPlane(glm::vec3 point, float radius, const Plane& plane, glm::vec3& normal) {
        float distance = glm::dot(point, plane.normal) - plane.distance;
        
        if (distance < radius) {
            normal = plane.normal;
            return true;
        }
        
        return false;
    }
};

#endif