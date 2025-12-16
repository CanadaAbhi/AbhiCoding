#ifndef TERRAIN_H
#define TERRAIN_H

#include <glm/glm.hpp>
#include <vector>
#include <cmath>
#include <random>

class Terrain {
public:
    int width;
    int depth;
    float cellSize;
    std::vector<std::vector<float>> heightMap;
    
    Terrain(int w, int d, float size = 10.0f)
        : width(w), depth(d), cellSize(size)
    {
        heightMap.resize(depth, std::vector<float>(width, 0.0f));
        GenerateHeightMap();
    }
    
    void GenerateHeightMap() {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(0.0, 1.0);
        
        // Simple noise-based terrain generation
        for (int z = 0; z < depth; z++) {
            for (int x = 0; x < width; x++) {
                float height = 0.0f;
                float frequency = 0.05f;
                float amplitude = 20.0f;
                
                // Multi-octave noise
                for (int octave = 0; octave < 4; octave++) {
                    float sampleX = x * frequency;
                    float sampleZ = z * frequency;
                    
                    float noiseValue = SimplexNoise(sampleX, sampleZ);
                    height += noiseValue * amplitude;
                    
                    frequency *= 2.0f;
                    amplitude *= 0.5f;
                }
                
                // Add some randomness
                height += (dis(gen) - 0.5f) * 2.0f;
                
                heightMap[z][x] = height;
            }
        }
        
        // Smooth the terrain
        SmoothTerrain(2);
    }
    
    void GenerateFlatRunway() {
        // Create a flat runway in the center
        int runwayLength = depth / 3;
        int runwayWidth = width / 20;
        int centerX = width / 2;
        int centerZ = depth / 2;
        
        for (int z = centerZ - runwayLength / 2; z < centerZ + runwayLength / 2; z++) {
            for (int x = centerX - runwayWidth; x < centerX + runwayWidth; x++) {
                if (z >= 0 && z < depth && x >= 0 && x < width) {
                    heightMap[z][x] = 0.5f;
                }
            }
        }
    }
    
    float GetHeight(float worldX, float worldZ) const {
        // Convert world coordinates to grid coordinates
        float gridX = worldX / cellSize + width / 2.0f;
        float gridZ = worldZ / cellSize + depth / 2.0f;
        
        // Check bounds
        if (gridX < 0 || gridX >= width - 1 || gridZ < 0 || gridZ >= depth - 1) {
            return 0.0f;
        }
        
        // Bilinear interpolation
        int x0 = static_cast<int>(gridX);
        int z0 = static_cast<int>(gridZ);
        int x1 = x0 + 1;
        int z1 = z0 + 1;
        
        float fx = gridX - x0;
        float fz = gridZ - z0;
        
        float h00 = heightMap[z0][x0];
        float h10 = heightMap[z0][x1];
        float h01 = heightMap[z1][x0];
        float h11 = heightMap[z1][x1];
        
        float h0 = h00 * (1 - fx) + h10 * fx;
        float h1 = h01 * (1 - fx) + h11 * fx;
        
        return h0 * (1 - fz) + h1 * fz;
    }
    
    glm::vec3 GetNormal(float worldX, float worldZ) const {
        // Sample nearby heights to calculate normal
        float offset = cellSize;
        
        float hL = GetHeight(worldX - offset, worldZ);
        float hR = GetHeight(worldX + offset, worldZ);
        float hD = GetHeight(worldX, worldZ - offset);
        float hU = GetHeight(worldX, worldZ + offset);
        
        glm::vec3 normal;
        normal.x = hL - hR;
        normal.y = 2.0f * offset;
        normal.z = hD - hU;
        
        return glm::normalize(normal);
    }
    
    glm::vec3 GetWorldPosition(int x, int z) const {
        float worldX = (x - width / 2.0f) * cellSize;
        float worldZ = (z - depth / 2.0f) * cellSize;
        float worldY = heightMap[z][x];
        return glm::vec3(worldX, worldY, worldZ);
    }
    
    void GetVerticesAndIndices(std::vector<float>& vertices, 
                              std::vector<unsigned int>& indices) const {
        vertices.clear();
        indices.clear();
        
        // Generate vertices
        for (int z = 0; z < depth; z++) {
            for (int x = 0; x < width; x++) {
                glm::vec3 pos = GetWorldPosition(x, z);
                glm::vec3 normal = GetNormal(pos.x, pos.z);
                
                // Position
                vertices.push_back(pos.x);
                vertices.push_back(pos.y);
                vertices.push_back(pos.z);
                
                // Normal
                vertices.push_back(normal.x);
                vertices.push_back(normal.y);
                vertices.push_back(normal.z);
                
                // Color based on height
                float colorValue = glm::clamp((pos.y + 10.0f) / 40.0f, 0.0f, 1.0f);
                vertices.push_back(0.2f + colorValue * 0.3f);
                vertices.push_back(0.5f + colorValue * 0.3f);
                vertices.push_back(0.2f + colorValue * 0.2f);
            }
        }
        
        // Generate indices
        for (int z = 0; z < depth - 1; z++) {
            for (int x = 0; x < width - 1; x++) {
                int topLeft = z * width + x;
                int topRight = topLeft + 1;
                int bottomLeft = (z + 1) * width + x;
                int bottomRight = bottomLeft + 1;
                
                // First triangle
                indices.push_back(topLeft);
                indices.push_back(bottomLeft);
                indices.push_back(topRight);
                
                // Second triangle
                indices.push_back(topRight);
                indices.push_back(bottomLeft);
                indices.push_back(bottomRight);
            }
        }
    }

private:
    // Simple 2D simplex-style noise
    float SimplexNoise(float x, float z) const {
        // Very simplified noise function
        float n = std::sin(x * 0.1f) * std::cos(z * 0.1f) +
                  std::sin(x * 0.3f + z * 0.2f) * 0.5f +
                  std::sin(x * 0.7f - z * 0.5f) * 0.25f;
        return n;
    }
    
    void SmoothTerrain(int iterations) {
        for (int iter = 0; iter < iterations; iter++) {
            std::vector<std::vector<float>> smoothed = heightMap;
            
            for (int z = 1; z < depth - 1; z++) {
                for (int x = 1; x < width - 1; x++) {
                    float sum = 0.0f;
                    int count = 0;
                    
                    for (int dz = -1; dz <= 1; dz++) {
                        for (int dx = -1; dx <= 1; dx++) {
                            sum += heightMap[z + dz][x + dx];
                            count++;
                        }
                    }
                    
                    smoothed[z][x] = sum / count;
                }
            }
            
            heightMap = smoothed;
        }
    }
};

#endif