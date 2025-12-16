#ifndef PLAYER_H
#define PLAYER_H

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include "Maze.h"

class Player {
public:
    glm::vec3 position;
    glm::vec3 front;
    glm::vec3 up;
    glm::vec3 right;
    
    float yaw;
    float pitch;
    float movementSpeed;
    float mouseSensitivity;
    float height;
    float radius;
    
    Player(glm::vec3 startPos = glm::vec3(1.0f, 1.0f, 1.0f))
        : position(startPos),
          front(glm::vec3(0.0f, 0.0f, -1.0f)),
          up(glm::vec3(0.0f, 1.0f, 0.0f)),
          yaw(-90.0f),
          pitch(0.0f),
          movementSpeed(5.0f),
          mouseSensitivity(0.1f),
          height(1.5f),
          radius(0.3f)
    {
        updateVectors();
    }
    
    glm::mat4 GetViewMatrix() const {
        return glm::lookAt(position, position + front, up);
    }
    
    void ProcessMouseMovement(float xoffset, float yoffset, bool constrainPitch = true) {
        xoffset *= mouseSensitivity;
        yoffset *= mouseSensitivity;
        
        yaw += xoffset;
        pitch += yoffset;
        
        if (constrainPitch) {
            if (pitch > 89.0f)
                pitch = 89.0f;
            if (pitch < -89.0f)
                pitch = -89.0f;
        }
        
        updateVectors();
    }
    
    void MoveForward(float deltaTime, const Maze& maze) {
        glm::vec3 newPos = position + front * movementSpeed * deltaTime;
        if (!CheckCollision(newPos, maze)) {
            position = newPos;
        }
    }
    
    void MoveBackward(float deltaTime, const Maze& maze) {
        glm::vec3 newPos = position - front * movementSpeed * deltaTime;
        if (!CheckCollision(newPos, maze)) {
            position = newPos;
        }
    }
    
    void MoveLeft(float deltaTime, const Maze& maze) {
        glm::vec3 newPos = position - right * movementSpeed * deltaTime;
        if (!CheckCollision(newPos, maze)) {
            position = newPos;
        }
    }
    
    void MoveRight(float deltaTime, const Maze& maze) {
        glm::vec3 newPos = position + right * movementSpeed * deltaTime;
        if (!CheckCollision(newPos, maze)) {
            position = newPos;
        }
    }
    
    // Get current grid position
    glm::ivec2 GetGridPosition() const {
        float cellSize = 2.0f;
        return glm::ivec2(
            static_cast<int>(position.x / cellSize),
            static_cast<int>(position.z / cellSize)
        );
    }
    
    // Check if player reached the goal
    bool ReachedGoal(const Maze& maze) const {
        glm::ivec2 gridPos = GetGridPosition();
        return gridPos.x == maze.endPos.x && gridPos.y == maze.endPos.y;
    }

private:
    void updateVectors() {
        glm::vec3 frontTemp;
        frontTemp.x = cos(glm::radians(yaw)) * cos(glm::radians(pitch));
        frontTemp.y = sin(glm::radians(pitch));
        frontTemp.z = sin(glm::radians(yaw)) * cos(glm::radians(pitch));
        front = glm::normalize(frontTemp);
        
        right = glm::normalize(glm::cross(front, glm::vec3(0.0f, 1.0f, 0.0f)));
        up = glm::normalize(glm::cross(right, front));
    }
    
    bool CheckCollision(glm::vec3 newPos, const Maze& maze) const {
        float cellSize = 2.0f;
        float wallThickness = 0.2f;
        float wallHeight = 2.0f;
        
        // Get grid cell
        int gridX = static_cast<int>(newPos.x / cellSize);
        int gridZ = static_cast<int>(newPos.z / cellSize);
        
        // Check bounds
        if (gridX < 0 || gridX >= maze.width || gridZ < 0 || gridZ >= maze.height) {
            return true;
        }
        
        // Get cell position in world space
        float cellWorldX = gridX * cellSize;
        float cellWorldZ = gridZ * cellSize;
        
        const Cell& cell = maze.grid[gridZ][gridX];
        
        // Check collision with each wall
        // North wall
        if (cell.walls[NORTH]) {
            float wallZ = cellWorldZ;
            if (newPos.z - radius < wallZ + wallThickness && 
                newPos.z + radius > wallZ &&
                newPos.x > cellWorldX && newPos.x < cellWorldX + cellSize) {
                return true;
            }
        }
        
        // South wall
        if (cell.walls[SOUTH]) {
            float wallZ = cellWorldZ + cellSize;
            if (newPos.z + radius > wallZ - wallThickness && 
                newPos.z - radius < wallZ &&
                newPos.x > cellWorldX && newPos.x < cellWorldX + cellSize) {
                return true;
            }
        }
        
        // West wall
        if (cell.walls[WEST]) {
            float wallX = cellWorldX;
            if (newPos.x - radius < wallX + wallThickness && 
                newPos.x + radius > wallX &&
                newPos.z > cellWorldZ && newPos.z < cellWorldZ + cellSize) {
                return true;
            }
        }
        
        // East wall
        if (cell.walls[EAST]) {
            float wallX = cellWorldX + cellSize;
            if (newPos.x + radius > wallX - wallThickness && 
                newPos.x - radius < wallX &&
                newPos.z > cellWorldZ && newPos.z < cellWorldZ + cellSize) {
                return true;
            }
        }
        
        return false;
    }
};

#endif