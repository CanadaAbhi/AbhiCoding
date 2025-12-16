#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "Aircraft.h"
#include "FlightCamera.h"
#include "Terrain.h"
#include "HUD.h"

#include <iostream>
#include <vector>

// Settings
const unsigned int SCR_WIDTH = 1600;
const unsigned int SCR_HEIGHT = 900;

// Objects
Aircraft aircraft;
Camera camera;
Terrain* terrain = nullptr;

// Input
float lastX = SCR_WIDTH / 2.0f;
float lastY = SCR_HEIGHT / 2.0f;
bool firstMouse = true;

// Timing
float deltaTime = 0.0f;
float lastFrame = 0.0f;

// UI
bool showHUD = true;

// Function prototypes
void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void mouse_callback(GLFWwindow* window, double xpos, double ypos);
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);
void processInput(GLFWwindow *window);
unsigned int compileShader(const char* vertexSource, const char* fragmentSource);
void renderTerrain(unsigned int shader, unsigned int VAO, int indexCount);
void renderAircraft(unsigned int shader);
void renderSkybox(unsigned int shader);
void renderHUD();

// Shaders
const char* vertexShaderSource = R"(
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;
layout (location = 2) in vec3 aColor;

out vec3 FragPos;
out vec3 Normal;
out vec3 Color;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main()
{
    FragPos = vec3(model * vec4(aPos, 1.0));
    Normal = mat3(transpose(inverse(model))) * aNormal;
    Color = aColor;
    gl_Position = projection * view * vec4(FragPos, 1.0);
}
)";

const char* fragmentShaderSource = R"(
#version 330 core
out vec4 FragColor;

in vec3 FragPos;
in vec3 Normal;
in vec3 Color;

uniform vec3 lightPos;
uniform vec3 viewPos;
uniform vec3 objectColor;
uniform bool useVertexColor;

void main()
{
    vec3 color = useVertexColor ? Color : objectColor;
    
    // Ambient
    float ambientStrength = 0.4;
    vec3 ambient = ambientStrength * color;
    
    // Diffuse
    vec3 norm = normalize(Normal);
    vec3 lightDir = normalize(lightPos - FragPos);
    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse = diff * color;
    
    // Specular
    float specularStrength = 0.3;
    vec3 viewDir = normalize(viewPos - FragPos);
    vec3 reflectDir = reflect(-lightDir, norm);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), 16);
    vec3 specular = specularStrength * spec * vec3(1.0);
    
    vec3 result = ambient + diffuse + specular;
    FragColor = vec4(result, 1.0);
}
)";

const char* skyboxVertexShader = R"(
#version 330 core
layout (location = 0) in vec3 aPos;

out vec3 TexCoords;

uniform mat4 projection;
uniform mat4 view;

void main()
{
    TexCoords = aPos;
    vec4 pos = projection * view * vec4(aPos, 1.0);
    gl_Position = pos.xyww;
}
)";

const char* skyboxFragmentShader = R"(
#version 330 core
out vec4 FragColor;

in vec3 TexCoords;

void main()
{
    // Simple gradient sky
    vec3 topColor = vec3(0.3, 0.5, 0.9);
    vec3 bottomColor = vec3(0.8, 0.9, 1.0);
    
    float t = (TexCoords.y + 1.0) * 0.5;
    vec3 color = mix(bottomColor, topColor, t);
    
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
    GLFWwindow* window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "Flight Simulator Basics", NULL, NULL);
    if (window == NULL) {
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    glfwSetCursorPosCallback(window, mouse_callback);
    glfwSetScrollCallback(window, scroll_callback);

    // Capture mouse in free camera mode
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);

    // Load OpenGL functions
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cout << "Failed to initialize GLAD" << std::endl;
        return -1;
    }

    // Configure OpenGL
    glEnable(GL_DEPTH_TEST);

    // Build shaders
    unsigned int shaderProgram = compileShader(vertexShaderSource, fragmentShaderSource);
    unsigned int skyboxShader = compileShader(skyboxVertexShader, skyboxFragmentShader);

    // Generate terrain
    std::cout << "Generating terrain..." << std::endl;
    terrain = new Terrain(100, 100, 10.0f);
    terrain->GenerateFlatRunway();
    
    std::vector<float> terrainVertices;
    std::vector<unsigned int> terrainIndices;
    terrain->GetVerticesAndIndices(terrainVertices, terrainIndices);
    
    // Setup terrain VAO
    unsigned int terrainVAO, terrainVBO, terrainEBO;
    glGenVertexArrays(1, &terrainVAO);
    glGenBuffers(1, &terrainVBO);
    glGenBuffers(1, &terrainEBO);
    
    glBindVertexArray(terrainVAO);
    
    glBindBuffer(GL_ARRAY_BUFFER, terrainVBO);
    glBufferData(GL_ARRAY_BUFFER, terrainVertices.size() * sizeof(float), 
                 terrainVertices.data(), GL_STATIC_DRAW);
    
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, terrainEBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, terrainIndices.size() * sizeof(unsigned int),
                 terrainIndices.data(), GL_STATIC_DRAW);
    
    // Position
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 9 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    // Normal
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 9 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);
    // Color
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 9 * sizeof(float), (void*)(6 * sizeof(float)));
    glEnableVertexAttribArray(2);
    
    // Setup skybox VAO
    float skyboxVertices[] = {
        -1.0f,  1.0f, -1.0f,
        -1.0f, -1.0f, -1.0f,
         1.0f, -1.0f, -1.0f,
         1.0f, -1.0f, -1.0f,
         1.0f,  1.0f, -1.0f,
        -1.0f,  1.0f, -1.0f,

        -1.0f, -1.0f,  1.0f,
        -1.0f, -1.0f, -1.0f,
        -1.0f,  1.0f, -1.0f,
        -1.0f,  1.0f, -1.0f,
        -1.0f,  1.0f,  1.0f,
        -1.0f, -1.0f,  1.0f,

         1.0f, -1.0f, -1.0f,
         1.0f, -1.0f,  1.0f,
         1.0f,  1.0f,  1.0f,
         1.0f,  1.0f,  1.0f,
         1.0f,  1.0f, -1.0f,
         1.0f, -1.0f, -1.0f,

        -1.0f, -1.0f,  1.0f,
        -1.0f,  1.0f,  1.0f,
         1.0f,  1.0f,  1.0f,
         1.0f,  1.0f,  1.0f,
         1.0f, -1.0f,  1.0f,
        -1.0f, -1.0f,  1.0f,

        -1.0f,  1.0f, -1.0f,
         1.0f,  1.0f, -1.0f,
         1.0f,  1.0f,  1.0f,
         1.0f,  1.0f,  1.0f,
        -1.0f,  1.0f,  1.0f,
        -1.0f,  1.0f, -1.0f,

        -1.0f, -1.0f, -1.0f,
        -1.0f, -1.0f,  1.0f,
         1.0f, -1.0f, -1.0f,
         1.0f, -1.0f, -1.0f,
        -1.0f, -1.0f,  1.0f,
         1.0f, -1.0f,  1.0f
    };
    
    unsigned int skyboxVAO, skyboxVBO;
    glGenVertexArrays(1, &skyboxVAO);
    glGenBuffers(1, &skyboxVBO);
    glBindVertexArray(skyboxVAO);
    glBindBuffer(GL_ARRAY_BUFFER, skyboxVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(skyboxVertices), skyboxVertices, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    
    // Initialize aircraft at runway
    aircraft.position = glm::vec3(0.0f, 10.0f, 0.0f);
    aircraft.groundHeight = terrain->GetHeight(0.0f, 0.0f);
    
    std::cout << "\n=== FLIGHT SIMULATOR CONTROLS ===" << std::endl;
    std::cout << "W/S - Pitch (Elevator)" << std::endl;
    std::cout << "A/D - Roll (Ailerons)" << std::endl;
    std::cout << "Q/E - Yaw (Rudder)" << std::endl;
    std::cout << "SHIFT - Increase Throttle" << std::endl;
    std::cout << "CTRL - Decrease Throttle" << std::endl;
    std::cout << "F - Flaps" << std::endl;
    std::cout << "B - Brake" << std::endl;
    std::cout << "C - Cycle Camera Mode" << std::endl;
    std::cout << "H - Toggle HUD" << std::endl;
    std::cout << "R - Reset Aircraft" << std::endl;
    std::cout << "ESC - Exit" << std::endl;
    std::cout << "================================\n" << std::endl;

    // Render loop
    while (!glfwWindowShouldClose(window)) {
        // Timing
        float currentFrame = glfwGetTime();
        deltaTime = currentFrame - lastFrame;
        lastFrame = currentFrame;

        // Input
        processInput(window);
        
        // Update aircraft physics
        aircraft.groundHeight = terrain->GetHeight(aircraft.position.x, aircraft.position.z);
        aircraft.Update(deltaTime);
        
        // Update camera
        camera.Update(aircraft, deltaTime);

        // Render
        glClearColor(0.5f, 0.7f, 1.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // View/Projection matrices
        glm::mat4 projection = glm::perspective(glm::radians(45.0f), 
                                                (float)SCR_WIDTH / (float)SCR_HEIGHT, 
                                                0.1f, 10000.0f);
        glm::mat4 view = camera.GetViewMatrix();
        
        // Render skybox first
        glDepthFunc(GL_LEQUAL);
        glUseProgram(skyboxShader);
        glm::mat4 skyView = glm::mat4(glm::mat3(view)); // Remove translation
        glUniformMatrix4fv(glGetUniformLocation(skyboxShader, "view"), 1, GL_FALSE, glm::value_ptr(skyView));
        glUniformMatrix4fv(glGetUniformLocation(skyboxShader, "projection"), 1, GL_FALSE, glm::value_ptr(projection));
        glBindVertexArray(skyboxVAO);
        glDrawArrays(GL_TRIANGLES, 0, 36);
        glDepthFunc(GL_LESS);
        
        // Render terrain
        glUseProgram(shaderProgram);
        glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "projection"), 1, GL_FALSE, glm::value_ptr(projection));
        glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "view"), 1, GL_FALSE, glm::value_ptr(view));
        glUniform3fv(glGetUniformLocation(shaderProgram, "lightPos"), 1, 
                     glm::value_ptr(glm::vec3(1000.0f, 2000.0f, 1000.0f)));
        glUniform3fv(glGetUniformLocation(shaderProgram, "viewPos"), 1, 
                     glm::value_ptr(camera.position));
        
        glm::mat4 model = glm::mat4(1.0f);
        glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "model"), 1, GL_FALSE, glm::value_ptr(model));
        glUniform1i(glGetUniformLocation(shaderProgram, "useVertexColor"), 1);
        
        glBindVertexArray(terrainVAO);
        glDrawElements(GL_TRIANGLES, terrainIndices.size(), GL_UNSIGNED_INT, 0);
        
        // Render aircraft (if not in cockpit view)
        if (camera.mode != COCKPIT) {
            glUniform1i(glGetUniformLocation(shaderProgram, "useVertexColor"), 0);
            renderAircraft(shaderProgram);
        }
        
        // Render HUD
        if (showHUD) {
            renderHUD();
        }

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // Cleanup
    glDeleteVertexArrays(1, &terrainVAO);
    glDeleteBuffers(1, &terrainVBO);
    glDeleteBuffers(1, &terrainEBO);
    glDeleteVertexArrays(1, &skyboxVAO);
    glDeleteBuffers(1, &skyboxVBO);
    glDeleteProgram(shaderProgram);
    glDeleteProgram(skyboxShader);
    
    delete terrain;
    
    glfwTerminate();
    return 0;
}

void processInput(GLFWwindow *window) {
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);

    // Flight controls
    // Pitch (elevator)
    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
        aircraft.pitch = glm::min(aircraft.pitch + deltaTime * 2.0f, 1.0f);
    else if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
        aircraft.pitch = glm::max(aircraft.pitch - deltaTime * 2.0f, -1.0f);
    else
        aircraft.pitch *= 0.95f; // Return to center
    
    // Roll (ailerons)
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
        aircraft.roll = glm::max(aircraft.roll - deltaTime * 2.0f, -1.0f);
    else if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
        aircraft.roll = glm::min(aircraft.roll + deltaTime * 2.0f, 1.0f);
    else
        aircraft.roll *= 0.95f; // Return to center
    
    // Yaw (rudder)
    if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS)
        aircraft.yaw = glm::max(aircraft.yaw - deltaTime * 2.0f, -1.0f);
    else if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS)
        aircraft.yaw = glm::min(aircraft.yaw + deltaTime * 2.0f, 1.0f);
    else
        aircraft.yaw *= 0.95f; // Return to center
    
    // Throttle
    if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS)
        aircraft.throttle += deltaTime * 0.5f;
    if (glfwGetKey(window, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS)
        aircraft.throttle -= deltaTime * 0.5f;
    
    // Flaps
    static bool fKeyPressed = false;
    if (glfwGetKey(window, GLFW_KEY_F) == GLFW_PRESS && !fKeyPressed) {
        aircraft.flaps = (aircraft.flaps > 0.5f) ? 0.0f : 1.0f;
        fKeyPressed = true;
    }
    if (glfwGetKey(window, GLFW_KEY_F) == GLFW_RELEASE)
        fKeyPressed = false;
    
    // Brake
    if (glfwGetKey(window, GLFW_KEY_B) == GLFW_PRESS)
        aircraft.brake = 1.0f;
    else
        aircraft.brake = 0.0f;
    
    // Camera controls
    static bool cKeyPressed = false;
    if (glfwGetKey(window, GLFW_KEY_C) == GLFW_PRESS && !cKeyPressed) {
        camera.CycleMode();
        cKeyPressed = true;
        
        const char* modeNames[] = {"COCKPIT", "CHASE", "ORBIT", "TOWER", "FREE"};
        std::cout << "Camera mode: " << modeNames[camera.mode] << std::endl;
    }
    if (glfwGetKey(window, GLFW_KEY_C) == GLFW_RELEASE)
        cKeyPressed = false;
    
    // HUD toggle
    static bool hKeyPressed = false;
    if (glfwGetKey(window, GLFW_KEY_H) == GLFW_PRESS && !hKeyPressed) {
        showHUD = !showHUD;
        hKeyPressed = true;
    }
    if (glfwGetKey(window, GLFW_KEY_H) == GLFW_RELEASE)
        hKeyPressed = false;
    
    // Reset
    static bool rKeyPressed = false;
    if (glfwGetKey(window, GLFW_KEY_R) == GLFW_PRESS && !rKeyPressed) {
        aircraft = Aircraft(glm::vec3(0.0f, 100.0f, 0.0f));
        aircraft.groundHeight = terrain->GetHeight(0.0f, 0.0f);
        rKeyPressed = true;
        std::cout << "Aircraft reset" << std::endl;
    }
    if (glfwGetKey(window, GLFW_KEY_R) == GLFW_RELEASE)
        rKeyPressed = false;
    
    // Free camera movement (when in free mode)
    if (camera.mode == FREE) {
        if (glfwGetKey(window, GLFW_KEY_UP) == GLFW_PRESS)
            camera.MoveForward(deltaTime);
        if (glfwGetKey(window, GLFW_KEY_DOWN) == GLFW_PRESS)
            camera.MoveBackward(deltaTime);
        if (glfwGetKey(window, GLFW_KEY_LEFT) == GLFW_PRESS)
            camera.MoveLeft(deltaTime);
        if (glfwGetKey(window, GLFW_KEY_RIGHT) == GLFW_PRESS)
            camera.MoveRight(deltaTime);
    }
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

    camera.ProcessMouseMovement(xoffset, yoffset);
}

void scroll_callback(GLFWwindow* window, double xoffset, double yoffset) {
    camera.AdjustDistance(static_cast<float>(-yoffset) * 2.0f);
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

void renderAircraft(unsigned int shader) {
    // Simple aircraft model (fuselage + wings)
    static unsigned int aircraftVAO = 0;
    
    if (aircraftVAO == 0) {
        float vertices[] = {
            // Fuselage (elongated box)
            // ... (simplified for brevity - use basic box geometry)
            -0.5f, -0.3f, -2.0f,  0.0f, -1.0f, 0.0f,
             0.5f, -0.3f, -2.0f,  0.0f, -1.0f, 0.0f,
             0.5f, -0.3f,  2.0f,  0.0f, -1.0f, 0.0f,
            -0.5f, -0.3f,  2.0f,  0.0f, -1.0f, 0.0f,
        };
        
        // Create simple aircraft VAO (placeholder)
        glGenVertexArrays(1, &aircraftVAO);
        // ... setup VAO
    }
    
    glm::mat4 model = aircraft.GetModelMatrix();
    glUniformMatrix4fv(glGetUniformLocation(shader, "model"), 1, GL_FALSE, glm::value_ptr(model));
    glUniform3f(glGetUniformLocation(shader, "objectColor"), 0.8f, 0.2f, 0.2f);
    
    // Render simple cube as aircraft placeholder
    // (In production, load actual 3D model)
}

void renderHUD() {
    // Console output HUD (for simplicity - in production use text rendering)
    static float hudUpdateTimer = 0.0f;
    hudUpdateTimer += deltaTime;
    
    if (hudUpdateTimer > 0.5f) {
        system("clear"); // or "cls" on Windows
        
        std::cout << "=== FLIGHT SIMULATOR ===" << std::endl;
        std::cout << HUD::GetSpeedString(aircraft) << "  " << HUD::GetAltitudeString(aircraft) << std::endl;
        std::cout << HUD::GetHeadingString(aircraft) << "  " << HUD::GetVerticalSpeedString(aircraft) << std::endl;
        std::cout << HUD::GetThrottleString(aircraft) << "  " << HUD::GetFlapsString(aircraft) << std::endl;
        std::cout << HUD::GetAttitudeString(aircraft) << std::endl;
        std::cout << HUD::GetAngleOfAttackString(aircraft) << "  " << HUD::GetGForceString(aircraft) << std::endl;
        std::cout << HUD::GetGroundStatusString(aircraft) << std::endl;
        std::cout << "=======================" << std::endl;
        
        hudUpdateTimer = 0.0f;
    }
}