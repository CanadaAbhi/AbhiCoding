#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "Maze.h"
#include "Player.h"

#include <iostream>
#include <sstream>

// Settings
const unsigned int SCR_WIDTH = 1280;
const unsigned int SCR_HEIGHT = 720;

// Player and maze
Player player;
Maze* currentMaze = nullptr;

// Camera
float lastX = SCR_WIDTH / 2.0f;
float lastY = SCR_HEIGHT / 2.0f;
bool firstMouse = true;

// Timing
float deltaTime = 0.0f;
float lastFrame = 0.0f;

// View mode
bool topDownView = false;
bool showSolution = false;
bool showMinimap = true;

// Function prototypes
void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void mouse_callback(GLFWwindow* window, double xpos, double ypos);
void processInput(GLFWwindow *window);
unsigned int compileShader(const char* vertexSource, const char* fragmentSource);
void renderMaze(const Maze& maze, unsigned int shaderProgram, bool showPath = false);
void renderFloor(const Maze& maze, unsigned int shaderProgram);
void renderCube();
void renderQuad();
void renderMinimap(const Maze& maze, unsigned int shaderProgram);

// Shader sources
const char* vertexShaderSource = R"(
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;

out vec3 FragPos;
out vec3 Normal;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main()
{
    FragPos = vec3(model * vec4(aPos, 1.0));
    Normal = mat3(transpose(inverse(model))) * aNormal;
    gl_Position = projection * view * vec4(FragPos, 1.0);
}
)";

const char* fragmentShaderSource = R"(
#version 330 core
out vec4 FragColor;

in vec3 FragPos;
in vec3 Normal;

uniform vec3 objectColor;
uniform vec3 lightPos;
uniform vec3 viewPos;
uniform float ambientStrength;

void main()
{
    // Ambient
    vec3 ambient = ambientStrength * vec3(1.0);
    
    // Diffuse
    vec3 norm = normalize(Normal);
    vec3 lightDir = normalize(lightPos - FragPos);
    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse = diff * vec3(1.0);
    
    // Specular
    float specularStrength = 0.3;
    vec3 viewDir = normalize(viewPos - FragPos);
    vec3 reflectDir = reflect(-lightDir, norm);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), 16);
    vec3 specular = specularStrength * spec * vec3(1.0);
    
    vec3 result = (ambient + diffuse + specular) * objectColor;
    FragColor = vec4(result, 1.0);
}
)";

const char* simpleVertexShader = R"(
#version 330 core
layout (location = 0) in vec3 aPos;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main()
{
    gl_Position = projection * view * model * vec4(aPos, 1.0);
}
)";

const char* simpleFragmentShader = R"(
#version 330 core
out vec4 FragColor;

uniform vec3 color;

void main()
{
    FragColor = vec4(color, 1.0);
}
)";

int main() {
    // Initialize GLFW
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

    // Create window
    GLFWwindow* window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "3D Maze Generator & Navigator", NULL, NULL);
    if (window == NULL) {
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    glfwSetCursorPosCallback(window, mouse_callback);

    // Capture mouse
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

    // Load OpenGL functions
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cout << "Failed to initialize GLAD" << std::endl;
        return -1;
    }

    // Configure OpenGL
    glEnable(GL_DEPTH_TEST);

    // Build and compile shaders
    unsigned int shaderProgram = compileShader(vertexShaderSource, fragmentShaderSource);
    unsigned int simpleShader = compileShader(simpleVertexShader, simpleFragmentShader);

    // Generate maze
    std::cout << "Generating maze..." << std::endl;
    currentMaze = new Maze(15, 15);
    currentMaze->GenerateRecursiveBacktracking();
    currentMaze->SolveAStar();
    
    // Set player at start position
    player = Player(glm::vec3(1.0f, 1.0f, 1.0f));
    
    std::cout << "Controls:" << std::endl;
    std::cout << "  WASD - Move" << std::endl;
    std::cout << "  Mouse - Look around" << std::endl;
    std::cout << "  T - Toggle top-down view" << std::endl;
    std::cout << "  S - Toggle solution path" << std::endl;
    std::cout << "  M - Toggle minimap" << std::endl;
    std::cout << "  R - Regenerate maze (Recursive Backtracking)" << std::endl;
    std::cout << "  P - Regenerate maze (Prim's Algorithm)" << std::endl;
    std::cout << "  1 - Solve with BFS" << std::endl;
    std::cout << "  2 - Solve with DFS" << std::endl;
    std::cout << "  3 - Solve with A*" << std::endl;
    std::cout << "  ESC - Exit" << std::endl;

    // Render loop
    while (!glfwWindowShouldClose(window)) {
        // Per-frame time logic
        float currentFrame = glfwGetTime();
        deltaTime = currentFrame - lastFrame;
        lastFrame = currentFrame;

        // Input
        processInput(window);

        // Render
        glClearColor(0.05f, 0.05f, 0.1f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // Setup view
        glm::mat4 projection;
        glm::mat4 view;
        
        if (topDownView) {
            // Top-down orthographic view
            float size = currentMaze->width * 1.5f;
            projection = glm::ortho(-size, size, -size, size, 0.1f, 100.0f);
            view = glm::lookAt(
                glm::vec3(currentMaze->width, 20.0f, currentMaze->height),
                glm::vec3(currentMaze->width, 0.0f, currentMaze->height),
                glm::vec3(0.0f, 0.0f, -1.0f)
            );
        } else {
            // First-person perspective
            projection = glm::perspective(glm::radians(45.0f), 
                                         (float)SCR_WIDTH / (float)SCR_HEIGHT, 
                                         0.1f, 100.0f);
            view = player.GetViewMatrix();
        }

        // Render maze
        glUseProgram(shaderProgram);
        glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "projection"), 1, GL_FALSE, glm::value_ptr(projection));
        glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "view"), 1, GL_FALSE, glm::value_ptr(view));
        glUniform3fv(glGetUniformLocation(shaderProgram, "lightPos"), 1, 
                     glm::value_ptr(glm::vec3(currentMaze->width, 10.0f, currentMaze->height)));
        glUniform3fv(glGetUniformLocation(shaderProgram, "viewPos"), 1, 
                     glm::value_ptr(player.position));

        renderFloor(*currentMaze, shaderProgram);
        renderMaze(*currentMaze, shaderProgram, showSolution);
        
        // Render start marker (green)
        glm::mat4 model = glm::mat4(1.0f);
        model = glm::translate(model, currentMaze->GetWorldPos(currentMaze->startPos.x, currentMaze->startPos.y) + 
                              glm::vec3(1.0f, 0.5f, 1.0f));
        model = glm::scale(model, glm::vec3(0.5f, 1.0f, 0.5f));
        glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "model"), 1, GL_FALSE, glm::value_ptr(model));
        glUniform3f(glGetUniformLocation(shaderProgram, "objectColor"), 0.2f, 1.0f, 0.2f);
        glUniform1f(glGetUniformLocation(shaderProgram, "ambientStrength"), 0.5f);
        renderCube();
        
        // Render end marker (red)
        model = glm::mat4(1.0f);
        model = glm::translate(model, currentMaze->GetWorldPos(currentMaze->endPos.x, currentMaze->endPos.y) + 
                              glm::vec3(1.0f, 0.5f, 1.0f));
        model = glm::scale(model, glm::vec3(0.5f, 1.0f, 0.5f));
        glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "model"), 1, GL_FALSE, glm::value_ptr(model));
        glUniform3f(glGetUniformLocation(shaderProgram, "objectColor"), 1.0f, 0.2f, 0.2f);
        renderCube();
        
        // Check if player reached goal
        if (player.ReachedGoal(*currentMaze)) {
            std::cout << "Congratulations! You reached the goal!" << std::endl;
            std::cout << "Press R or P to generate a new maze." << std::endl;
        }
        
        // Render minimap
        if (showMinimap && !topDownView) {
            renderMinimap(*currentMaze, simpleShader);
        }

        // Swap buffers and poll events
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // Cleanup
    delete currentMaze;
    glDeleteProgram(shaderProgram);
    glDeleteProgram(simpleShader);
    glfwTerminate();
    return 0;
}

void processInput(GLFWwindow *window) {
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);

    // Movement
    if (!topDownView) {
        if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
            player.MoveForward(deltaTime, *currentMaze);
        if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
            player.MoveBackward(deltaTime, *currentMaze);
        if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
            player.MoveLeft(deltaTime, *currentMaze);
        if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
            player.MoveRight(deltaTime, *currentMaze);
    }
    
    // Toggle views
    static bool tKeyPressed = false;
    if (glfwGetKey(window, GLFW_KEY_T) == GLFW_PRESS && !tKeyPressed) {
        topDownView = !topDownView;
        tKeyPressed = true;
    }
    if (glfwGetKey(window, GLFW_KEY_T) == GLFW_RELEASE)
        tKeyPressed = false;
    
    static bool sKeyPressed = false;
    if (glfwGetKey(window, GLFW_KEY_O) == GLFW_PRESS && !sKeyPressed) {
        showSolution = !showSolution;
        sKeyPressed = true;
    }
    if (glfwGetKey(window, GLFW_KEY_O) == GLFW_RELEASE)
        sKeyPressed = false;
    
    static bool mKeyPressed = false;
    if (glfwGetKey(window, GLFW_KEY_M) == GLFW_PRESS && !mKeyPressed) {
        showMinimap = !showMinimap;
        mKeyPressed = true;
    }
    if (glfwGetKey(window, GLFW_KEY_M) == GLFW_RELEASE)
        mKeyPressed = false;
    
    // Regenerate maze
    static bool rKeyPressed = false;
    if (glfwGetKey(window, GLFW_KEY_R) == GLFW_PRESS && !rKeyPressed) {
        delete currentMaze;
        currentMaze = new Maze(15, 15);
        currentMaze->GenerateRecursiveBacktracking();
        currentMaze->SolveAStar();
        player = Player(glm::vec3(1.0f, 1.0f, 1.0f));
        rKeyPressed = true;
        std::cout << "New maze generated (Recursive Backtracking)" << std::endl;
    }
    if (glfwGetKey(window, GLFW_KEY_R) == GLFW_RELEASE)
        rKeyPressed = false;
    
    static bool pKeyPressed = false;
    if (glfwGetKey(window, GLFW_KEY_P) == GLFW_PRESS && !pKeyPressed) {
        delete currentMaze;
        currentMaze = new Maze(15, 15);
        currentMaze->GeneratePrims();
        currentMaze->SolveAStar();
        player = Player(glm::vec3(1.0f, 1.0f, 1.0f));
        pKeyPressed = true;
        std::cout << "New maze generated (Prim's Algorithm)" << std::endl;
    }
    if (glfwGetKey(window, GLFW_KEY_P) == GLFW_RELEASE)
        pKeyPressed = false;
    
    // Solve maze
    static bool key1Pressed = false;
    if (glfwGetKey(window, GLFW_KEY_1) == GLFW_PRESS && !key1Pressed) {
        currentMaze->SolveBFS();
        showSolution = true;
        key1Pressed = true;
        std::cout << "Solved with BFS - Path length: " << currentMaze->solutionPath.size() << std::endl;
    }
    if (glfwGetKey(window, GLFW_KEY_1) == GLFW_RELEASE)
        key1Pressed = false;
    
    static bool key2Pressed = false;
    if (glfwGetKey(window, GLFW_KEY_2) == GLFW_PRESS && !key2Pressed) {
        currentMaze->SolveDFS();
        showSolution = true;
        key2Pressed = true;
        std::cout << "Solved with DFS - Path length: " << currentMaze->solutionPath.size() << std::endl;
    }
    if (glfwGetKey(window, GLFW_KEY_2) == GLFW_RELEASE)
        key2Pressed = false;
    
    static bool key3Pressed = false;
    if (glfwGetKey(window, GLFW_KEY_3) == GLFW_PRESS && !key3Pressed) {
        currentMaze->SolveAStar();
        showSolution = true;
        key3Pressed = true;
        std::cout << "Solved with A* - Path length: " << currentMaze->solutionPath.size() << std::endl;
    }
    if (glfwGetKey(window, GLFW_KEY_3) == GLFW_RELEASE)
        key3Pressed = false;
}

void framebuffer_size_callback(GLFWwindow* window, int width, int height) {
    glViewport(0, 0, width, height);
}

void mouse_callback(GLFWwindow* window, double xposIn, double yposIn) {
    float xpos = static_cast<float>(xposIn);
    float ypos = static_cast<float>(yposIn);

    if (firstMouse) {
        lastX = xpos;
        lastY = ypos;
        firstMouse = false;
    }

    float xoffset = xpos - lastX;
    float yoffset = lastY - ypos;

    lastX = xpos;
    lastY = ypos;

    if (!topDownView) {
        player.ProcessMouseMovement(xoffset, yoffset);
    }
}

unsigned int compileShader(const char* vertexSource, const char* fragmentSource) {
    unsigned int vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &vertexSource, NULL);
    glCompileShader(vertexShader);
    
    int success;
    char infoLog[512];
    glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
    if (!success) {
        glGetShaderInfoLog(vertexShader, 512, NULL, infoLog);
        std::cout << "ERROR::SHADER::VERTEX::COMPILATION_FAILED\n" << infoLog << std::endl;
    }
    
    unsigned int fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fragmentSource, NULL);
    glCompileShader(fragmentShader);
    
    glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);
    if (!success) {
        glGetShaderInfoLog(fragmentShader, 512, NULL, infoLog);
        std::cout << "ERROR::SHADER::FRAGMENT::COMPILATION_FAILED\n" << infoLog << std::endl;
    }
    
    unsigned int shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);
    
    glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);
    if (!success) {
        glGetProgramInfoLog(shaderProgram, 512, NULL, infoLog);
        std::cout << "ERROR::SHADER::PROGRAM::LINKING_FAILED\n" << infoLog << std::endl;
    }
    
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);
    
    return shaderProgram;
}

void renderMaze(const Maze& maze, unsigned int shaderProgram, bool showPath) {
    float cellSize = 2.0f;
    float wallHeight = 2.0f;
    float wallThickness = 0.2f;
    
    for (int y = 0; y < maze.height; y++) {
        for (int x = 0; x < maze.width; x++) {
            const Cell& cell = maze.grid[y][x];
            glm::vec3 cellPos = maze.GetWorldPos(x, y);
            
            // Wall color
            glm::vec3 wallColor(0.6f, 0.6f, 0.7f);
            if (showPath && cell.inPath) {
                wallColor = glm::vec3(0.3f, 0.3f, 1.0f); // Blue for path cells
            }
            
            // Render walls
            glUniform3fv(glGetUniformLocation(shaderProgram, "objectColor"), 1, glm::value_ptr(wallColor));
            glUniform1f(glGetUniformLocation(shaderProgram, "ambientStrength"), 0.3f);
            
            // North wall
            if (cell.walls[NORTH]) {
                glm::mat4 model = glm::mat4(1.0f);
                model = glm::translate(model, cellPos + glm::vec3(cellSize/2, wallHeight/2, 0.0f));
                model = glm::scale(model, glm::vec3(cellSize, wallHeight, wallThickness));
                glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "model"), 1, GL_FALSE, glm::value_ptr(model));
                renderCube();
            }
            
            // South wall
            if (cell.walls[SOUTH]) {
                glm::mat4 model = glm::mat4(1.0f);
                model = glm::translate(model, cellPos + glm::vec3(cellSize/2, wallHeight/2, cellSize));
                model = glm::scale(model, glm::vec3(cellSize, wallHeight, wallThickness));
                glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "model"), 1, GL_FALSE, glm::value_ptr(model));
                renderCube();
            }
            
            // West wall
            if (cell.walls[WEST]) {
                glm::mat4 model = glm::mat4(1.0f);
                model = glm::translate(model, cellPos + glm::vec3(0.0f, wallHeight/2, cellSize/2));
                model = glm::scale(model, glm::vec3(wallThickness, wallHeight, cellSize));
                glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "model"), 1, GL_FALSE, glm::value_ptr(model));
                renderCube();
            }
            
            // East wall
            if (cell.walls[EAST]) {
                glm::mat4 model = glm::mat4(1.0f);
                model = glm::translate(model, cellPos + glm::vec3(cellSize, wallHeight/2, cellSize/2));
                model = glm::scale(model, glm::vec3(wallThickness, wallHeight, cellSize));
                glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "model"), 1, GL_FALSE, glm::value_ptr(model));
                renderCube();
            }
            
            // Render path markers if solution is shown
            if (showPath && cell.inPath) {
                glm::mat4 model = glm::mat4(1.0f);
                model = glm::translate(model, cellPos + glm::vec3(1.0f, 0.05f, 1.0f));
                model = glm::scale(model, glm::vec3(0.3f, 0.1f, 0.3f));
                glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "model"), 1, GL_FALSE, glm::value_ptr(model));
                glUniform3f(glGetUniformLocation(shaderProgram, "objectColor"), 1.0f, 1.0f, 0.3f);
                glUniform1f(glGetUniformLocation(shaderProgram, "ambientStrength"), 0.8f);
                renderCube();
            }
        }
    }
}

void renderFloor(const Maze& maze, unsigned int shaderProgram) {
    float cellSize = 2.0f;
    
    glm::mat4 model = glm::mat4(1.0f);
    model = glm::translate(model, glm::vec3(maze.width * cellSize / 2, 0.0f, maze.height * cellSize / 2));
    model = glm::scale(model, glm::vec3(maze.width * cellSize, 0.1f, maze.height * cellSize));
    glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "model"), 1, GL_FALSE, glm::value_ptr(model));
    glUniform3f(glGetUniformLocation(shaderProgram, "objectColor"), 0.2f, 0.2f, 0.25f);
    glUniform1f(glGetUniformLocation(shaderProgram, "ambientStrength"), 0.5f);
    renderCube();
}

void renderMinimap(const Maze& maze, unsigned int shaderProgram) {
    // Save current viewport
    GLint viewport[4];
    glGetIntegerv(GL_VIEWPORT, viewport);
    
    // Set minimap viewport (top-right corner)
    int minimapSize = 200;
    glViewport(SCR_WIDTH - minimapSize - 10, SCR_HEIGHT - minimapSize - 10, minimapSize, minimapSize);
    
    glUseProgram(shaderProgram);
    
    // Orthographic projection for minimap
    glm::mat4 projection = glm::ortho(0.0f, (float)maze.width, 0.0f, (float)maze.height, -1.0f, 1.0f);
    glm::mat4 view = glm::mat4(1.0f);
    
    glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "projection"), 1, GL_FALSE, glm::value_ptr(projection));
    glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "view"), 1, GL_FALSE, glm::value_ptr(view));
    
    // Draw maze cells
    for (int y = 0; y < maze.height; y++) {
        for (int x = 0; x < maze.width; x++) {
            glm::mat4 model = glm::mat4(1.0f);
            model = glm::translate(model, glm::vec3(x + 0.5f, y + 0.5f, 0.0f));
            model = glm::scale(model, glm::vec3(0.9f, 0.9f, 1.0f));
            
            glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "model"), 1, GL_FALSE, glm::value_ptr(model));
            
            // Color based on cell state
            if (maze.grid[y][x].inPath && showSolution) {
                glUniform3f(glGetUniformLocation(shaderProgram, "color"), 1.0f, 1.0f, 0.3f);
            } else {
                glUniform3f(glGetUniformLocation(shaderProgram, "color"), 0.3f, 0.3f, 0.3f);
            }
            
            renderQuad();
        }
    }
    
    // Draw player position
    glm::ivec2 playerGrid = player.GetGridPosition();
    glm::mat4 model = glm::mat4(1.0f);
    model = glm::translate(model, glm::vec3(playerGrid.x + 0.5f, playerGrid.y + 0.5f, 0.0f));
    model = glm::scale(model, glm::vec3(0.5f, 0.5f, 1.0f));
    glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "model"), 1, GL_FALSE, glm::value_ptr(model));
    glUniform3f(glGetUniformLocation(shaderProgram, "color"), 0.2f, 1.0f, 0.2f);
    renderQuad();
    
    // Restore viewport
    glViewport(viewport[0], viewport[1], viewport[2], viewport[3]);
}

unsigned int cubeVAO = 0;
unsigned int cubeVBO;
void renderCube() {
    if (cubeVAO == 0) {
        float vertices[] = {
            -0.5f, -0.5f, -0.5f,  0.0f,  0.0f, -1.0f,
             0.5f, -0.5f, -0.5f,  0.0f,  0.0f, -1.0f,
             0.5f,  0.5f, -0.5f,  0.0f,  0.0f, -1.0f,
             0.5f,  0.5f, -0.5f,  0.0f,  0.0f, -1.0f,
            -0.5f,  0.5f, -0.5f,  0.0f,  0.0f, -1.0f,
            -0.5f, -0.5f, -0.5f,  0.0f,  0.0f, -1.0f,

            -0.5f, -0.5f,  0.5f,  0.0f,  0.0f,  1.0f,
             0.5f,  0.5f,  0.5f,  0.0f,  0.0f,  1.0f,
             0.5f, -0.5f,  0.5f,  0.0f,  0.0f,  1.0f,
             0.5f,  0.5f,  0.5f,  0.0f,  0.0f,  1.0f,
            -0.5f, -0.5f,  0.5f,  0.0f,  0.0f,  1.0f,
            -0.5f,  0.5f,  0.5f,  0.0f,  0.0f,  1.0f,

            -0.5f,  0.5f,  0.5f, -1.0f,  0.0f,  0.0f,
            -0.5f, -0.5f, -0.5f, -1.0f,  0.0f,  0.0f,
            -0.5f,  0.5f, -0.5f, -1.0f,  0.0f,  0.0f,
            -0.5f, -0.5f, -0.5f, -1.0f,  0.0f,  0.0f,
            -0.5f,  0.5f,  0.5f, -1.0f,  0.0f,  0.0f,
            -0.5f, -0.5f,  0.5f, -1.0f,  0.0f,  0.0f,

             0.5f,  0.5f,  0.5f,  1.0f,  0.0f,  0.0f,
             0.5f,  0.5f, -0.5f,  1.0f,  0.0f,  0.0f,
             0.5f, -0.5f, -0.5f,  1.0f,  0.0f,  0.0f,
             0.5f, -0.5f, -0.5f,  1.0f,  0.0f,  0.0f,
             0.5f, -0.5f,  0.5f,  1.0f,  0.0f,  0.0f,
             0.5f,  0.5f,  0.5f,  1.0f,  0.0f,  0.0f,

            -0.5f, -0.5f, -0.5f,  0.0f, -1.0f,  0.0f,
             0.5f, -0.5f,  0.5f,  0.0f, -1.0f,  0.0f,
             0.5f, -0.5f, -0.5f,  0.0f, -1.0f,  0.0f,
             0.5f, -0.5f,  0.5f,  0.0f, -1.0f,  0.0f,
            -0.5f, -0.5f, -0.5f,  0.0f, -1.0f,  0.0f,
            -0.5f, -0.5f,  0.5f,  0.0f, -1.0f,  0.0f,

            -0.5f,  0.5f, -0.5f,  0.0f,  1.0f,  0.0f,
             0.5f,  0.5f, -0.5f,  0.0f,  1.0f,  0.0f,
             0.5f,  0.5f,  0.5f,  0.0f,  1.0f,  0.0f,
             0.5f,  0.5f,  0.5f,  0.0f,  1.0f,  0.0f,
            -0.5f,  0.5f,  0.5f,  0.0f,  1.0f,  0.0f,
            -0.5f,  0.5f, -0.5f,  0.0f,  1.0f,  0.0f
        };
        
        glGenVertexArrays(1, &cubeVAO);
        glGenBuffers(1, &cubeVBO);
        
        glBindBuffer(GL_ARRAY_BUFFER, cubeVBO);
        glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
        
        glBindVertexArray(cubeVAO);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));
        glEnableVertexAttribArray(1);
    }
    
    glBindVertexArray(cubeVAO);
    glDrawArrays(GL_TRIANGLES, 0, 36);
    glBindVertexArray(0);
}

unsigned int quadVAO = 0;
unsigned int quadVBO;
void renderQuad() {
    if (quadVAO == 0) {
        float vertices[] = {
            -0.5f, -0.5f, 0.0f,
             0.5f, -0.5f, 0.0f,
             0.5f,  0.5f, 0.0f,
             0.5f,  0.5f, 0.0f,
            -0.5f,  0.5f, 0.0f,
            -0.5f, -0.5f, 0.0f
        };
        
        glGenVertexArrays(1, &quadVAO);
        glGenBuffers(1, &quadVBO);
        
        glBindBuffer(GL_ARRAY_BUFFER, quadVBO);
        glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
        
        glBindVertexArray(quadVAO);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
        glEnableVertexAttribArray(0);
    }
    
    glBindVertexArray(quadVAO);
    glDrawArrays(GL_TRIANGLES, 0, 6);
    glBindVertexArray(0);
}