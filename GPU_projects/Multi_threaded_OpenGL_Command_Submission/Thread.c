#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <time.h>

// Forward declare window
GLFWwindow* window = NULL;

typedef struct {
    GLuint vao;
    GLuint shader;
    GLuint texture;
    GLuint indexCount;
} DrawCommand;

// Thread Safe command Queue (fixed typo: Qeueue -> Queue)

#define MAX_COMMANDS 100000

typedef struct {
    DrawCommand commands[MAX_COMMANDS];
    int count;
    pthread_mutex_t mutex;
} CommandQueue;

CommandQueue gQueue;

void initQueue() {
    gQueue.count = 0;
    pthread_mutex_init(&gQueue.mutex, NULL);
}

void pushCommand(DrawCommand cmd) {
    pthread_mutex_lock(&gQueue.mutex);
    if (gQueue.count < MAX_COMMANDS) {
        gQueue.commands[gQueue.count++] = cmd;
    }
    pthread_mutex_unlock(&gQueue.mutex);
}

// Worker Threads generate commands

void* workerThread(void* arg) {
    int threadID = *(int*)arg;

    for (int i = 0; i < 5000; i++) {
        DrawCommand cmd;
        cmd.vao = 1;
        cmd.shader = 1;
        cmd.texture = threadID % 4;
        cmd.indexCount = 36;

        pushCommand(cmd);
    }
    return NULL;
}

// Render Thread: OpenGL Submission

void submitCommands() {
    glUseProgram(1);

    pthread_mutex_lock(&gQueue.mutex);
    int count = gQueue.count;
    DrawCommand* cmds = gQueue.commands;
    gQueue.count = 0;
    pthread_mutex_unlock(&gQueue.mutex);

    for (int i = 0; i < count; i++) {
        glBindVertexArray(cmds[i].vao);
        glBindTexture(GL_TEXTURE_2D, cmds[i].texture);
        glDrawElements(GL_TRIANGLES, cmds[i].indexCount, GL_UNSIGNED_INT, 0);
    }
}

// Timing CPU vs Driver Cost

double now() {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

void renderFrame() {
    double t0 = now();
    submitCommands();
    glFinish(); // force driver sync
    double t1 = now();

    printf("Submit+GPU sync: %.3f ms\n", (t1 - t0) * 1000.0);
}

// Initialize OpenGL resources
void initGL() {
    // Create a simple VAO
    GLuint vao;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    
    // Create simple VBO with cube vertices
    GLfloat vertices[] = {
        -0.5f, -0.5f, -0.5f,
         0.5f, -0.5f, -0.5f,
         0.5f,  0.5f, -0.5f,
        -0.5f,  0.5f, -0.5f,
        -0.5f, -0.5f,  0.5f,
         0.5f, -0.5f,  0.5f,
         0.5f,  0.5f,  0.5f,
        -0.5f,  0.5f,  0.5f,
    };
    
    GLuint vbo;
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
    
    // Create EBO with cube indices
    GLuint indices[] = {
        0, 1, 2, 2, 3, 0,
        1, 5, 6, 6, 2, 1,
        5, 4, 7, 7, 6, 5,
        4, 0, 3, 3, 7, 4,
        3, 2, 6, 6, 7, 3,
        4, 5, 1, 1, 0, 4
    };
    
    GLuint ebo;
    glGenBuffers(1, &ebo);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);
    
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    
    // Create simple textures
    for (int i = 0; i < 4; i++) {
        GLuint texture;
        glGenTextures(1, &texture);
        glBindTexture(GL_TEXTURE_2D, texture);
        
        unsigned char color[4] = {
            (unsigned char)(i * 64), 
            (unsigned char)(255 - i * 64), 
            (unsigned char)(128), 
            255
        };
        
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 1, 1, 0, GL_RGBA, GL_UNSIGNED_BYTE, color);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    }
    
    // Create a simple shader program (program ID will be 1)
    const char* vertexShaderSource = 
        "#version 330 core\n"
        "layout (location = 0) in vec3 aPos;\n"
        "void main() {\n"
        "   gl_Position = vec4(aPos, 1.0);\n"
        "}\n";
    
    const char* fragmentShaderSource = 
        "#version 330 core\n"
        "out vec4 FragColor;\n"
        "uniform sampler2D tex;\n"
        "void main() {\n"
        "   FragColor = texture(tex, vec2(0.5, 0.5));\n"
        "}\n";
    
    GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &vertexShaderSource, NULL);
    glCompileShader(vertexShader);
    
    GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fragmentShaderSource, NULL);
    glCompileShader(fragmentShader);
    
    GLuint shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);
    
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);
    
    glUseProgram(shaderProgram);
}

// Main program

#define NUM_WORKERS 4

int main() {
    // Init GLFW
    if (!glfwInit()) {
        fprintf(stderr, "Failed to initialize GLFW\n");
        return -1;
    }
    
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    
    window = glfwCreateWindow(800, 600, "Multithreaded Renderer", NULL, NULL);
    if (!window) {
        fprintf(stderr, "Failed to create GLFW window\n");
        glfwTerminate();
        return -1;
    }
    
    glfwMakeContextCurrent(window);
    
    // Init GLEW
    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) {
        fprintf(stderr, "Failed to initialize GLEW\n");
        return -1;
    }
    
    // Initialize OpenGL resources
    initGL();
    
    initQueue();

    pthread_t threads[NUM_WORKERS];
    int ids[NUM_WORKERS];

    for (int i = 0; i < NUM_WORKERS; i++) {
        ids[i] = i;
        pthread_create(&threads[i], NULL, workerThread, &ids[i]);
    }

    for (int i = 0; i < NUM_WORKERS; i++)
        pthread_join(threads[i], NULL);

    printf("All worker threads completed. Starting render loop...\n");

    while (!glfwWindowShouldClose(window)) {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        
        renderFrame();
        
        glfwSwapBuffers(window);
        glfwPollEvents();
    }
    
    pthread_mutex_destroy(&gQueue.mutex);
    glfwTerminate();
    
    return 0;
}