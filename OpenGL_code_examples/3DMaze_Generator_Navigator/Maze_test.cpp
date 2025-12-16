#include "Maze.h"
#include <iostream>
#include <iomanip>
#include <chrono>

// Console colors (ANSI escape codes)
#define RESET   "\033[0m"
#define RED     "\033[31m"
#define GREEN   "\033[32m"
#define YELLOW  "\033[33m"
#define BLUE    "\033[34m"
#define MAGENTA "\033[35m"
#define CYAN    "\033[36m"
#define WHITE   "\033[37m"

void printMaze(const Maze& maze, bool showSolution = false) {
    std::cout << "\n";
    
    // Top border
    for (int x = 0; x < maze.width; x++) {
        std::cout << "█████";
    }
    std::cout << "█\n";
    
    for (int y = 0; y < maze.height; y++) {
        // First line of cell (north walls and content)
        for (int x = 0; x < maze.width; x++) {
            std::cout << "█";
            if (maze.grid[y][x].walls[NORTH]) {
                std::cout << "████";
            } else {
                std::cout << "    ";
            }
        }
        std::cout << "█\n";
        
        // Second line of cell (west wall, content, east wall hint)
        for (int x = 0; x < maze.width; x++) {
            // West wall
            if (maze.grid[y][x].walls[WEST]) {
                std::cout << "█";
            } else {
                std::cout << " ";
            }
            
            // Cell content
            if (x == maze.startPos.x && y == maze.startPos.y) {
                std::cout << GREEN << " S  " << RESET;
            } else if (x == maze.endPos.x && y == maze.endPos.y) {
                std::cout << RED << " E  " << RESET;
            } else if (showSolution && maze.grid[y][x].inPath) {
                std::cout << YELLOW << " *  " << RESET;
            } else if (showSolution && maze.grid[y][x].explored) {
                std::cout << CYAN << " .  " << RESET;
            } else {
                std::cout << "    ";
            }
        }
        
        // East border
        std::cout << "█\n";
        
        // Third line (prepare for south walls)
        for (int x = 0; x < maze.width; x++) {
            std::cout << "█";
            if (maze.grid[y][x].walls[SOUTH] && y == maze.height - 1) {
                std::cout << "████";
            } else if (y < maze.height - 1 && maze.grid[y][x].walls[SOUTH]) {
                // Will be handled by next row
                std::cout << "    ";
            } else {
                std::cout << "    ";
            }
        }
        std::cout << "█\n";
    }
    
    // Bottom border
    for (int x = 0; x < maze.width; x++) {
        std::cout << "█████";
    }
    std::cout << "█\n\n";
    
    // Legend
    std::cout << "Legend: " << GREEN << "S" << RESET << " = Start, " 
              << RED << "E" << RESET << " = End";
    if (showSolution) {
        std::cout << ", " << YELLOW << "*" << RESET << " = Solution path, "
                  << CYAN << "." << RESET << " = Explored";
    }
    std::cout << "\n\n";
}

void printSimpleMaze(const Maze& maze, bool showSolution = false) {
    std::cout << "\n";
    
    // Top border
    std::cout << " ";
    for (int x = 0; x < maze.width; x++) {
        std::cout << "___";
    }
    std::cout << "\n";
    
    for (int y = 0; y < maze.height; y++) {
        std::cout << "|";
        for (int x = 0; x < maze.width; x++) {
            // Cell content
            if (x == maze.startPos.x && y == maze.startPos.y) {
                std::cout << GREEN << "S" << RESET;
            } else if (x == maze.endPos.x && y == maze.endPos.y) {
                std::cout << RED << "E" << RESET;
            } else if (showSolution && maze.grid[y][x].inPath) {
                std::cout << YELLOW << "*" << RESET;
            } else if (showSolution && maze.grid[y][x].explored) {
                std::cout << CYAN << "." << RESET;
            } else {
                std::cout << " ";
            }
            
            // East wall
            if (maze.grid[y][x].walls[EAST]) {
                std::cout << "|";
            } else {
                std::cout << " ";
            }
            
            // South wall
            if (y < maze.height - 1 && maze.grid[y][x].walls[SOUTH]) {
                std::cout << "_";
            } else {
                std::cout << " ";
            }
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}

void benchmarkAlgorithm(const std::string& name, std::function<void(Maze&)> func, Maze& maze, int iterations = 100) {
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < iterations; i++) {
        func(maze);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    double avgTime = duration.count() / (double)iterations / 1000.0; // Convert to milliseconds
    
    std::cout << std::setw(25) << std::left << name 
              << std::setw(12) << std::right << std::fixed << std::setprecision(3) 
              << avgTime << " ms" << std::endl;
}

int main() {
    std::cout << "==============================================\n";
    std::cout << "   3D Maze Generator - Algorithm Tester\n";
    std::cout << "==============================================\n\n";
    
    int mazeSize = 15;
    
    // Test Recursive Backtracking
    std::cout << CYAN << "TEST 1: Recursive Backtracking Generation" << RESET << "\n";
    std::cout << "-------------------------------------------\n";
    Maze maze1(mazeSize, mazeSize);
    auto start = std::chrono::high_resolution_clock::now();
    maze1.GenerateRecursiveBacktracking();
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "Generation time: " << duration.count() / 1000.0 << " ms\n";
    printSimpleMaze(maze1);
    
    // Test Prim's Algorithm
    std::cout << CYAN << "TEST 2: Prim's Algorithm Generation" << RESET << "\n";
    std::cout << "-------------------------------------------\n";
    Maze maze2(mazeSize, mazeSize);
    start = std::chrono::high_resolution_clock::now();
    maze2.GeneratePrims();
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "Generation time: " << duration.count() / 1000.0 << " ms\n";
    printSimpleMaze(maze2);
    
    // Test BFS Pathfinding
    std::cout << CYAN << "TEST 3: Breadth-First Search (BFS)" << RESET << "\n";
    std::cout << "-------------------------------------------\n";
    start = std::chrono::high_resolution_clock::now();
    bool foundBFS = maze1.SolveBFS();
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "Search time: " << duration.count() / 1000.0 << " ms\n";
    std::cout << "Solution found: " << (foundBFS ? GREEN "YES" : RED "NO") << RESET << "\n";
    std::cout << "Path length: " << maze1.solutionPath.size() << " cells\n";
    printSimpleMaze(maze1, true);
    
    // Test DFS Pathfinding
    std::cout << CYAN << "TEST 4: Depth-First Search (DFS)" << RESET << "\n";
    std::cout << "-------------------------------------------\n";
    start = std::chrono::high_resolution_clock::now();
    bool foundDFS = maze1.SolveDFS();
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "Search time: " << duration.count() / 1000.0 << " ms\n";
    std::cout << "Solution found: " << (foundDFS ? GREEN "YES" : RED "NO") << RESET << "\n";
    std::cout << "Path length: " << maze1.solutionPath.size() << " cells\n";
    printSimpleMaze(maze1, true);
    
    // Test A* Pathfinding
    std::cout << CYAN << "TEST 5: A* Search" << RESET << "\n";
    std::cout << "-------------------------------------------\n";
    start = std::chrono::high_resolution_clock::now();
    bool foundAStar = maze1.SolveAStar();
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "Search time: " << duration.count() / 1000.0 << " ms\n";
    std::cout << "Solution found: " << (foundAStar ? GREEN "YES" : RED "NO") << RESET << "\n";
    std::cout << "Path length: " << maze1.solutionPath.size() << " cells\n";
    printSimpleMaze(maze1, true);
    
    // Benchmark comparison
    std::cout << CYAN << "\nPERFORMANCE BENCHMARKS (100 iterations)" << RESET << "\n";
    std::cout << "==============================================\n";
    std::cout << std::setw(25) << std::left << "Algorithm" 
              << std::setw(12) << std::right << "Avg Time" << std::endl;
    std::cout << "----------------------------------------------\n";
    
    Maze benchMaze(mazeSize, mazeSize);
    
    benchmarkAlgorithm("Recursive Backtracking", 
        [](Maze& m) { m = Maze(15, 15); m.GenerateRecursiveBacktracking(); }, 
        benchMaze);
    
    benchmarkAlgorithm("Prim's Algorithm", 
        [](Maze& m) { m = Maze(15, 15); m.GeneratePrims(); }, 
        benchMaze);
    
    benchMaze.GenerateRecursiveBacktracking();
    
    benchmarkAlgorithm("BFS Solve", 
        [](Maze& m) { m.SolveBFS(); }, 
        benchMaze);
    
    benchmarkAlgorithm("DFS Solve", 
        [](Maze& m) { m.SolveDFS(); }, 
        benchMaze);
    
    benchmarkAlgorithm("A* Solve", 
        [](Maze& m) { m.SolveAStar(); }, 
        benchMaze);
    
    std::cout << "==============================================\n";
    
    // Size comparison
    std::cout << CYAN << "\nSCALABILITY TEST" << RESET << "\n";
    std::cout << "==============================================\n";
    std::cout << std::setw(10) << "Size" 
              << std::setw(15) << "Generation" 
              << std::setw(15) << "A* Solve" << std::endl;
    std::cout << "----------------------------------------------\n";
    
    int sizes[] = {5, 10, 15, 20, 25, 30};
    for (int size : sizes) {
        Maze testMaze(size, size);
        
        start = std::chrono::high_resolution_clock::now();
        testMaze.GenerateRecursiveBacktracking();
        end = std::chrono::high_resolution_clock::now();
        auto genTime = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        start = std::chrono::high_resolution_clock::now();
        testMaze.SolveAStar();
        end = std::chrono::high_resolution_clock::now();
        auto solveTime = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        std::cout << std::setw(10) << (std::to_string(size) + "x" + std::to_string(size))
                  << std::setw(12) << std::fixed << std::setprecision(2) 
                  << genTime.count() / 1000.0 << " ms"
                  << std::setw(12) << solveTime.count() / 1000.0 << " ms" << std::endl;
    }
    
    std::cout << "==============================================\n\n";
    
    // Algorithm comparison on same maze
    std::cout << CYAN << "PATHFINDING COMPARISON (Same Maze)" << RESET << "\n";
    std::cout << "==============================================\n";
    Maze compMaze(20, 20);
    compMaze.GenerateRecursiveBacktracking();
    
    compMaze.SolveBFS();
    int bfsLength = compMaze.solutionPath.size();
    
    compMaze.SolveDFS();
    int dfsLength = compMaze.solutionPath.size();
    
    compMaze.SolveAStar();
    int astarLength = compMaze.solutionPath.size();
    
    std::cout << "Maze: 20x20\n";
    std::cout << "BFS path length:   " << std::setw(4) << bfsLength << " (optimal)\n";
    std::cout << "DFS path length:   " << std::setw(4) << dfsLength 
              << (dfsLength == bfsLength ? " (optimal)" : " (suboptimal)") << "\n";
    std::cout << "A* path length:    " << std::setw(4) << astarLength << " (optimal)\n";
    std::cout << "==============================================\n\n";
    
    std::cout << GREEN << "All tests completed successfully!\n" << RESET;
    std::cout << "Compile with OpenGL to see the 3D visualization.\n\n";
    
    return 0;
}