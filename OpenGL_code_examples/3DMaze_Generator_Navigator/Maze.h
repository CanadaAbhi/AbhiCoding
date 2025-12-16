#ifndef MAZE_H
#define MAZE_H

#include <vector>
#include <stack>
#include <queue>
#include <random>
#include <algorithm>
#include <glm/glm.hpp>

// Cell structure for maze
struct Cell {
    bool visited;
    bool walls[4]; // North, East, South, West
    bool inPath;   // Part of solution path
    bool explored; // Explored during pathfinding
    
    Cell() : visited(false), inPath(false), explored(false) {
        walls[0] = walls[1] = walls[2] = walls[3] = true;
    }
};

// Direction enum
enum Direction {
    NORTH = 0,
    EAST = 1,
    SOUTH = 2,
    WEST = 3
};

class Maze {
public:
    int width;
    int height;
    std::vector<std::vector<Cell>> grid;
    
    glm::vec2 startPos;
    glm::vec2 endPos;
    
    std::vector<glm::vec2> solutionPath;
    
    // Constructor
    Maze(int w, int h) : width(w), height(h) {
        grid.resize(height, std::vector<Cell>(width));
        startPos = glm::vec2(0, 0);
        endPos = glm::vec2(width - 1, height - 1);
    }
    
    // Generate maze using Recursive Backtracking
    void GenerateRecursiveBacktracking() {
        std::random_device rd;
        std::mt19937 gen(rd());
        
        std::stack<glm::ivec2> stack;
        glm::ivec2 current(0, 0);
        grid[0][0].visited = true;
        
        while (true) {
            std::vector<Direction> neighbors = GetUnvisitedNeighbors(current);
            
            if (!neighbors.empty()) {
                // Choose random neighbor
                std::uniform_int_distribution<> dis(0, neighbors.size() - 1);
                Direction dir = neighbors[dis(gen)];
                
                glm::ivec2 next = GetNeighborPos(current, dir);
                
                // Remove walls
                RemoveWall(current, next);
                
                grid[next.y][next.x].visited = true;
                stack.push(current);
                current = next;
            } else if (!stack.empty()) {
                current = stack.top();
                stack.pop();
            } else {
                break;
            }
        }
    }
    
    // Generate maze using Prim's Algorithm
    void GeneratePrims() {
        std::random_device rd;
        std::mt19937 gen(rd());
        
        std::vector<glm::ivec2> walls;
        glm::ivec2 start(0, 0);
        grid[0][0].visited = true;
        
        AddWallsToList(start, walls);
        
        while (!walls.empty()) {
            std::uniform_int_distribution<> dis(0, walls.size() - 1);
            int idx = dis(gen);
            glm::ivec2 wall = walls[idx];
            walls.erase(walls.begin() + idx);
            
            // Find the two cells separated by this wall
            std::vector<glm::ivec2> adjacentCells = GetAdjacentCells(wall);
            
            if (adjacentCells.size() == 2) {
                glm::ivec2 cell1 = adjacentCells[0];
                glm::ivec2 cell2 = adjacentCells[1];
                
                bool c1Visited = IsValid(cell1) && grid[cell1.y][cell1.x].visited;
                bool c2Visited = IsValid(cell2) && grid[cell2.y][cell2.x].visited;
                
                if (c1Visited != c2Visited) {
                    RemoveWall(cell1, cell2);
                    
                    glm::ivec2 newCell = c1Visited ? cell2 : cell1;
                    grid[newCell.y][newCell.x].visited = true;
                    AddWallsToList(newCell, walls);
                }
            }
        }
    }
    
    // Solve maze using BFS (Breadth-First Search)
    bool SolveBFS() {
        ClearSolution();
        
        std::queue<glm::ivec2> queue;
        std::vector<std::vector<glm::ivec2>> parent(height, std::vector<glm::ivec2>(width, glm::ivec2(-1, -1)));
        
        glm::ivec2 start(startPos.x, startPos.y);
        glm::ivec2 end(endPos.x, endPos.y);
        
        queue.push(start);
        grid[start.y][start.x].explored = true;
        
        while (!queue.empty()) {
            glm::ivec2 current = queue.front();
            queue.pop();
            
            if (current == end) {
                // Reconstruct path
                ReconstructPath(parent, start, end);
                return true;
            }
            
            // Check all four directions
            for (int dir = 0; dir < 4; dir++) {
                if (!grid[current.y][current.x].walls[dir]) {
                    glm::ivec2 next = GetNeighborPos(current, (Direction)dir);
                    
                    if (IsValid(next) && !grid[next.y][next.x].explored) {
                        grid[next.y][next.x].explored = true;
                        parent[next.y][next.x] = current;
                        queue.push(next);
                    }
                }
            }
        }
        
        return false;
    }
    
    // Solve maze using DFS (Depth-First Search)
    bool SolveDFS() {
        ClearSolution();
        
        std::stack<glm::ivec2> stack;
        std::vector<std::vector<glm::ivec2>> parent(height, std::vector<glm::ivec2>(width, glm::ivec2(-1, -1)));
        
        glm::ivec2 start(startPos.x, startPos.y);
        glm::ivec2 end(endPos.x, endPos.y);
        
        stack.push(start);
        grid[start.y][start.x].explored = true;
        
        while (!stack.empty()) {
            glm::ivec2 current = stack.top();
            stack.pop();
            
            if (current == end) {
                ReconstructPath(parent, start, end);
                return true;
            }
            
            for (int dir = 0; dir < 4; dir++) {
                if (!grid[current.y][current.x].walls[dir]) {
                    glm::ivec2 next = GetNeighborPos(current, (Direction)dir);
                    
                    if (IsValid(next) && !grid[next.y][next.x].explored) {
                        grid[next.y][next.x].explored = true;
                        parent[next.y][next.x] = current;
                        stack.push(next);
                    }
                }
            }
        }
        
        return false;
    }
    
    // Solve using A* algorithm
    bool SolveAStar() {
        ClearSolution();
        
        struct Node {
            glm::ivec2 pos;
            float g; // Cost from start
            float h; // Heuristic to end
            float f; // Total cost
            
            bool operator>(const Node& other) const {
                return f > other.f;
            }
        };
        
        std::priority_queue<Node, std::vector<Node>, std::greater<Node>> openSet;
        std::vector<std::vector<glm::ivec2>> parent(height, std::vector<glm::ivec2>(width, glm::ivec2(-1, -1)));
        std::vector<std::vector<float>> gScore(height, std::vector<float>(width, INFINITY));
        
        glm::ivec2 start(startPos.x, startPos.y);
        glm::ivec2 end(endPos.x, endPos.y);
        
        gScore[start.y][start.x] = 0;
        openSet.push({start, 0, Heuristic(start, end), Heuristic(start, end)});
        
        while (!openSet.empty()) {
            Node current = openSet.top();
            openSet.pop();
            
            if (grid[current.pos.y][current.pos.x].explored) continue;
            grid[current.pos.y][current.pos.x].explored = true;
            
            if (current.pos == end) {
                ReconstructPath(parent, start, end);
                return true;
            }
            
            for (int dir = 0; dir < 4; dir++) {
                if (!grid[current.pos.y][current.pos.x].walls[dir]) {
                    glm::ivec2 next = GetNeighborPos(current.pos, (Direction)dir);
                    
                    if (IsValid(next)) {
                        float tentativeG = gScore[current.pos.y][current.pos.x] + 1.0f;
                        
                        if (tentativeG < gScore[next.y][next.x]) {
                            parent[next.y][next.x] = current.pos;
                            gScore[next.y][next.x] = tentativeG;
                            float h = Heuristic(next, end);
                            openSet.push({next, tentativeG, h, tentativeG + h});
                        }
                    }
                }
            }
        }
        
        return false;
    }
    
    // Check if position is valid
    bool IsValid(glm::ivec2 pos) const {
        return pos.x >= 0 && pos.x < width && pos.y >= 0 && pos.y < height;
    }
    
    // Get 3D position from grid coordinates
    glm::vec3 GetWorldPos(int x, int y) const {
        float cellSize = 2.0f;
        return glm::vec3(x * cellSize, 0.0f, y * cellSize);
    }

private:
    // Get unvisited neighbors
    std::vector<Direction> GetUnvisitedNeighbors(glm::ivec2 pos) {
        std::vector<Direction> neighbors;
        
        // North
        if (pos.y > 0 && !grid[pos.y - 1][pos.x].visited)
            neighbors.push_back(NORTH);
        // East
        if (pos.x < width - 1 && !grid[pos.y][pos.x + 1].visited)
            neighbors.push_back(EAST);
        // South
        if (pos.y < height - 1 && !grid[pos.y + 1][pos.x].visited)
            neighbors.push_back(SOUTH);
        // West
        if (pos.x > 0 && !grid[pos.y][pos.x - 1].visited)
            neighbors.push_back(WEST);
        
        return neighbors;
    }
    
    // Get neighbor position based on direction
    glm::ivec2 GetNeighborPos(glm::ivec2 pos, Direction dir) const {
        switch (dir) {
            case NORTH: return glm::ivec2(pos.x, pos.y - 1);
            case EAST:  return glm::ivec2(pos.x + 1, pos.y);
            case SOUTH: return glm::ivec2(pos.x, pos.y + 1);
            case WEST:  return glm::ivec2(pos.x - 1, pos.y);
        }
        return pos;
    }
    
    // Remove wall between two cells
    void RemoveWall(glm::ivec2 current, glm::ivec2 next) {
        int dx = next.x - current.x;
        int dy = next.y - current.y;
        
        if (dx == 1) {
            grid[current.y][current.x].walls[EAST] = false;
            grid[next.y][next.x].walls[WEST] = false;
        } else if (dx == -1) {
            grid[current.y][current.x].walls[WEST] = false;
            grid[next.y][next.x].walls[EAST] = false;
        } else if (dy == 1) {
            grid[current.y][current.x].walls[SOUTH] = false;
            grid[next.y][next.x].walls[NORTH] = false;
        } else if (dy == -1) {
            grid[current.y][current.x].walls[NORTH] = false;
            grid[next.y][next.x].walls[SOUTH] = false;
        }
    }
    
    // Add walls to list for Prim's algorithm
    void AddWallsToList(glm::ivec2 cell, std::vector<glm::ivec2>& walls) {
        for (int dir = 0; dir < 4; dir++) {
            glm::ivec2 neighbor = GetNeighborPos(cell, (Direction)dir);
            if (IsValid(neighbor)) {
                walls.push_back(neighbor);
            }
        }
    }
    
    // Get cells adjacent to a wall
    std::vector<glm::ivec2> GetAdjacentCells(glm::ivec2 wall) {
        std::vector<glm::ivec2> cells;
        
        for (int dir = 0; dir < 4; dir++) {
            glm::ivec2 neighbor = GetNeighborPos(wall, (Direction)dir);
            if (IsValid(neighbor)) {
                cells.push_back(neighbor);
            }
        }
        
        return cells;
    }
    
    // Manhattan distance heuristic
    float Heuristic(glm::ivec2 a, glm::ivec2 b) const {
        return abs(a.x - b.x) + abs(a.y - b.y);
    }
    
    // Reconstruct path from parent array
    void ReconstructPath(const std::vector<std::vector<glm::ivec2>>& parent, 
                        glm::ivec2 start, glm::ivec2 end) {
        solutionPath.clear();
        glm::ivec2 current = end;
        
        while (current != start) {
            grid[current.y][current.x].inPath = true;
            solutionPath.push_back(glm::vec2(current.x, current.y));
            current = parent[current.y][current.x];
        }
        
        grid[start.y][start.x].inPath = true;
        solutionPath.push_back(glm::vec2(start.x, start.y));
        std::reverse(solutionPath.begin(), solutionPath.end());
    }
    
    // Clear solution
    void ClearSolution() {
        solutionPath.clear();
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                grid[y][x].inPath = false;
                grid[y][x].explored = false;
            }
        }
    }
};

#endif