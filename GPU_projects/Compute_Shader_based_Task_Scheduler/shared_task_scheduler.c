/*
 * shader_based_task_scheduler.c - GPU Compute Shader Task Scheduler
 * 
 * Demonstrates:
 * - Compute shader parallel processing
 * - SSBO (Shader Storage Buffer Objects)
 * - Dynamic workgroup dispatch
 * - Particle physics simulation on GPU
 * - Visualization of compute results
 */

 #include <stdio.h>
 #include <stdlib.h>
 #include <math.h>
 #include <string.h>
 #include <time.h>
 
 #include <GL/glew.h>
 #include <GLFW/glfw3.h>
 
 #define NUM_PARTICLES 10000
 #define WORKGROUP_SIZE 128
 #define WINDOW_WIDTH 1200
 #define WINDOW_HEIGHT 800
 
 typedef struct {
     float x, y, z, w;       // position.xyz, w unused
     float vx, vy, vz, vw;   // velocity.xyz, w unused
 } Particle;
 
 // Global OpenGL objects
 GLuint computeProgram;
 GLuint renderProgram;
 GLuint ssbo;
 GLuint vao, vbo;
 
 // Performance tracking
 double lastTime = 0.0;
 int frameCount = 0;
 double computeTime = 0.0;
 
 // Error checking helper
 void checkCompileErrors(GLuint shader, const char* type) {
     GLint success;
     GLchar infoLog[1024];
     if (strcmp(type, "PROGRAM") != 0) {
         glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
         if (!success) {
             glGetShaderInfoLog(shader, 1024, NULL, infoLog);
             printf("ERROR::SHADER_COMPILATION_ERROR of type: %s\n%s\n", type, infoLog);
         }
     } else {
         glGetProgramiv(shader, GL_LINK_STATUS, &success);
         if (!success) {
             glGetProgramInfoLog(shader, 1024, NULL, infoLog);
             printf("ERROR::PROGRAM_LINKING_ERROR of type: %s\n%s\n", type, infoLog);
         }
     }
 }
 
 // Compute shader source - runs on GPU
 const char* computeShaderSrc = "#version 430 core\n"
 "layout(local_size_x = 128) in;\n"
 "\n"
 "struct Particle {\n"
 "    vec4 pos;\n"
 "    vec4 vel;\n"
 "};\n"
 "\n"
 "layout(std430, binding = 0) buffer Particles {\n"
 "    Particle particles[];\n"
 "};\n"
 "\n"
 "uniform float dt;\n"
 "uniform float time;\n"
 "\n"
 "void main() {\n"
 "    uint idx = gl_GlobalInvocationID.x;\n"
 "    if (idx >= particles.length()) return;\n"
 "    \n"
 "    // Simple physics integration\n"
 "    particles[idx].pos.xyz += particles[idx].vel.xyz * dt;\n"
 "    \n"
 "    // Add gravity\n"
 "    particles[idx].vel.y -= 0.0001;\n"
 "    \n"
 "    // Bounce off boundaries [-1,1]\n"
 "    for (int i = 0; i < 3; i++) {\n"
 "        if (particles[idx].pos[i] > 1.0) {\n"
 "            particles[idx].pos[i] = 1.0;\n"
 "            particles[idx].vel[i] *= -0.8;\n"  // Energy loss on bounce
 "        } else if (particles[idx].pos[i] < -1.0) {\n"
 "            particles[idx].pos[i] = -1.0;\n"
 "            particles[idx].vel[i] *= -0.8;\n"
 "        }\n"
 "    }\n"
 "}\n";
 
 // Vertex shader for rendering particles
 const char* vertexShaderSrc = "#version 330 core\n"
 "layout (location = 0) in vec4 aPos;\n"
 "layout (location = 1) in vec4 aVel;\n"
 "\n"
 "out vec3 particleColor;\n"
 "\n"
 "void main() {\n"
 "    gl_Position = vec4(aPos.xyz, 1.0);\n"
 "    gl_PointSize = 3.0;\n"
 "    \n"
 "    // Color based on velocity\n"
 "    float speed = length(aVel.xyz);\n"
 "    particleColor = vec3(speed * 50.0, 0.5, 1.0 - speed * 50.0);\n"
 "}\n";
 
 // Fragment shader for rendering particles
 const char* fragmentShaderSrc = "#version 330 core\n"
 "in vec3 particleColor;\n"
 "out vec4 FragColor;\n"
 "\n"
 "void main() {\n"
 "    // Round point sprites\n"
 "    vec2 coord = gl_PointCoord - vec2(0.5);\n"
 "    if (length(coord) > 0.5)\n"
 "        discard;\n"
 "    \n"
 "    FragColor = vec4(particleColor, 1.0);\n"
 "}\n";
 
 // Compile a shader
 GLuint compileShader(GLenum type, const char* src) {
     GLuint shader = glCreateShader(type);
     glShaderSource(shader, 1, &src, NULL);
     glCompileShader(shader);
     
     const char* typeName = (type == GL_COMPUTE_SHADER) ? "COMPUTE_SHADER" :
                            (type == GL_VERTEX_SHADER) ? "VERTEX_SHADER" :
                            "FRAGMENT_SHADER";
     checkCompileErrors(shader, typeName);
     
     return shader;
 }
 
 // Create compute shader program
 GLuint createComputeProgram() {
     GLuint shader = compileShader(GL_COMPUTE_SHADER, computeShaderSrc);
     GLuint program = glCreateProgram();
     glAttachShader(program, shader);
     glLinkProgram(program);
     checkCompileErrors(program, "PROGRAM");
     glDeleteShader(shader);
     return program;
 }
 
 // Create render shader program
 GLuint createRenderProgram() {
     GLuint vertShader = compileShader(GL_VERTEX_SHADER, vertexShaderSrc);
     GLuint fragShader = compileShader(GL_FRAGMENT_SHADER, fragmentShaderSrc);
     
     GLuint program = glCreateProgram();
     glAttachShader(program, vertShader);
     glAttachShader(program, fragShader);
     glLinkProgram(program);
     checkCompileErrors(program, "PROGRAM");
     
     glDeleteShader(vertShader);
     glDeleteShader(fragShader);
     
     return program;
 }
 
 // Initialize particles with random positions and velocities
 void initParticles(Particle* particles, int count) {
     srand((unsigned int)time(NULL));
     
     for (int i = 0; i < count; i++) {
         // Random position in box
         particles[i].x = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
         particles[i].y = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
         particles[i].z = ((float)rand() / RAND_MAX) * 0.5f - 0.25f;
         particles[i].w = 1.0f;
         
         // Random velocity
         particles[i].vx = ((float)rand() / RAND_MAX) * 0.02f - 0.01f;
         particles[i].vy = ((float)rand() / RAND_MAX) * 0.02f - 0.01f;
         particles[i].vz = ((float)rand() / RAND_MAX) * 0.01f - 0.005f;
         particles[i].vw = 0.0f;
     }
     
     printf("Initialized %d particles\n", count);
 }
 
 // Print GPU and compute shader info
 void printSystemInfo() {
     printf("╔════════════════════════════════════════════════════════╗\n");
     printf("║     GPU Compute Shader Task Scheduler                 ║\n");
     printf("╚════════════════════════════════════════════════════════╝\n\n");
     
     printf("Renderer: %s\n", glGetString(GL_RENDERER));
     printf("OpenGL Version: %s\n", glGetString(GL_VERSION));
     printf("GLSL Version: %s\n", glGetString(GL_SHADING_LANGUAGE_VERSION));
     
     // Get compute shader limits
     GLint maxWorkGroupCount[3];
     GLint maxWorkGroupSize[3];
     GLint maxWorkGroupInvocations;
     
     glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_COUNT, 0, &maxWorkGroupCount[0]);
     glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_COUNT, 1, &maxWorkGroupCount[1]);
     glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_COUNT, 2, &maxWorkGroupCount[2]);
     
     glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_SIZE, 0, &maxWorkGroupSize[0]);
     glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_SIZE, 1, &maxWorkGroupSize[1]);
     glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_SIZE, 2, &maxWorkGroupSize[2]);
     
     glGetIntegerv(GL_MAX_COMPUTE_WORK_GROUP_INVOCATIONS, &maxWorkGroupInvocations);
     
     printf("\nCompute Shader Limits:\n");
     printf("  Max Work Group Count: %d x %d x %d\n", 
            maxWorkGroupCount[0], maxWorkGroupCount[1], maxWorkGroupCount[2]);
     printf("  Max Work Group Size: %d x %d x %d\n",
            maxWorkGroupSize[0], maxWorkGroupSize[1], maxWorkGroupSize[2]);
     printf("  Max Work Group Invocations: %d\n", maxWorkGroupInvocations);
     
     printf("\nSimulation Parameters:\n");
     printf("  Particles: %d\n", NUM_PARTICLES);
     printf("  Work Group Size: %d\n", WORKGROUP_SIZE);
     printf("  Work Groups Dispatched: %d\n", 
            (NUM_PARTICLES + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE);
     printf("\n");
 }
 
 // Read back particle data for debugging
 void readbackParticles(int numToShow) {
     Particle* particles = (Particle*)malloc(sizeof(Particle) * NUM_PARTICLES);
     
     glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo);
     glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, 
                        sizeof(Particle) * NUM_PARTICLES, particles);
     
     printf("\nSample Particle Data (first %d):\n", numToShow);
     for (int i = 0; i < numToShow && i < NUM_PARTICLES; i++) {
         printf("  Particle %d: pos(%.3f, %.3f, %.3f) vel(%.4f, %.4f, %.4f)\n",
                i, particles[i].x, particles[i].y, particles[i].z,
                particles[i].vx, particles[i].vy, particles[i].vz);
     }
     
     free(particles);
 }
 
 // Calculate and display FPS
 void updateFPS(GLFWwindow* window) {
     double currentTime = glfwGetTime();
     frameCount++;
     
     if (currentTime - lastTime >= 1.0) {
         double fps = frameCount / (currentTime - lastTime);
         double msPerFrame = 1000.0 / fps;
         
         char title[256];
         snprintf(title, sizeof(title), 
                 "GPU Task Scheduler | FPS: %.1f (%.2f ms) | Compute: %.2f ms | Particles: %d",
                 fps, msPerFrame, computeTime * 1000.0, NUM_PARTICLES);
         glfwSetWindowTitle(window, title);
         
         frameCount = 0;
         lastTime = currentTime;
     }
 }
 
 int main() {
     // Initialize GLFW
     if (!glfwInit()) {
         fprintf(stderr, "Failed to initialize GLFW\n");
         return -1;
     }
     
     // Request OpenGL 4.3 core profile (needed for compute shaders)
     glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
     glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
     glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
     glfwWindowHint(GLFW_SAMPLES, 4); // MSAA
     
     // Create window
     GLFWwindow* window = glfwCreateWindow(WINDOW_WIDTH, WINDOW_HEIGHT, 
                                           "GPU Compute Shader Task Scheduler", 
                                           NULL, NULL);
     if (!window) {
         fprintf(stderr, "Failed to create GLFW window\n");
         fprintf(stderr, "Note: Compute shaders require OpenGL 4.3+\n");
         glfwTerminate();
         return -1;
     }
     
     glfwMakeContextCurrent(window);
     glfwSwapInterval(1); // Enable vsync
     
     // Initialize GLEW
     glewExperimental = GL_TRUE;
     if (glewInit() != GLEW_OK) {
         fprintf(stderr, "Failed to initialize GLEW\n");
         glfwTerminate();
         return -1;
     }
     
     // Check compute shader support
     if (!GLEW_ARB_compute_shader) {
         fprintf(stderr, "ERROR: Compute shaders not supported!\n");
         fprintf(stderr, "Your GPU/driver must support OpenGL 4.3 or GL_ARB_compute_shader\n");
         glfwTerminate();
         return -1;
     }
     
     // Print system info
     printSystemInfo();
     
     // Create shader programs
     computeProgram = createComputeProgram();
     renderProgram = createRenderProgram();
     
     // Initialize particle data
     Particle* particles = (Particle*)malloc(sizeof(Particle) * NUM_PARTICLES);
     initParticles(particles, NUM_PARTICLES);
     
     // Create SSBO (Shader Storage Buffer Object)
     glGenBuffers(1, &ssbo);
     glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo);
     glBufferData(GL_SHADER_STORAGE_BUFFER, 
                  sizeof(Particle) * NUM_PARTICLES, 
                  particles, 
                  GL_DYNAMIC_DRAW);
     glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ssbo);
     
     free(particles);
     
     // Create VAO for rendering
     glGenVertexArrays(1, &vao);
     glBindVertexArray(vao);
     
     // Bind SSBO as vertex buffer for rendering
     glBindBuffer(GL_ARRAY_BUFFER, ssbo);
     
     // Position attribute (vec4)
     glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, sizeof(Particle), (void*)0);
     glEnableVertexAttribArray(0);
     
     // Velocity attribute (vec4)
     glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, sizeof(Particle), 
                          (void*)(4 * sizeof(float)));
     glEnableVertexAttribArray(1);
     
     // OpenGL state
     glEnable(GL_PROGRAM_POINT_SIZE);
     glEnable(GL_BLEND);
     glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
     glClearColor(0.05f, 0.05f, 0.1f, 1.0f);
     
     float dt = 0.016f; // ~60 FPS timestep
     lastTime = glfwGetTime();
     
     printf("Starting simulation...\n");
     printf("Press ESC to exit\n\n");
     
     // Read initial state
     readbackParticles(5);
     
     // Main loop
     while (!glfwWindowShouldClose(window)) {
         double frameStart = glfwGetTime();
         
         // COMPUTE PHASE: Update particles on GPU
         double computeStart = glfwGetTime();
         
         glUseProgram(computeProgram);
         glUniform1f(glGetUniformLocation(computeProgram, "dt"), dt);
         glUniform1f(glGetUniformLocation(computeProgram, "time"), (float)glfwGetTime());
         
         // Calculate number of work groups needed
         int numGroups = (NUM_PARTICLES + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
         
         // Dispatch compute shader
         glDispatchCompute(numGroups, 1, 1);
         
         // Wait for compute shader to finish
         glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT);
         
         computeTime = glfwGetTime() - computeStart;
         
         // RENDER PHASE: Draw particles
         glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
         
         glUseProgram(renderProgram);
         glBindVertexArray(vao);
         glDrawArrays(GL_POINTS, 0, NUM_PARTICLES);
         
         // Update FPS counter
         updateFPS(window);
         
         // Swap buffers and poll events
         glfwSwapBuffers(window);
         glfwPollEvents();
         
         // Handle ESC key
         if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
             glfwSetWindowShouldClose(window, 1);
         }
         
         // Debug: Print particle data periodically
         static double lastPrint = 0.0;
         if (glfwGetTime() - lastPrint > 5.0) {
             readbackParticles(3);
             lastPrint = glfwGetTime();
         }
     }
     
     printf("\nCleaning up...\n");
     
     // Cleanup
     glDeleteProgram(computeProgram);
     glDeleteProgram(renderProgram);
     glDeleteBuffers(1, &ssbo);
     glDeleteVertexArrays(1, &vao);
     
     glfwDestroyWindow(window);
     glfwTerminate();
     
     printf("Shutdown complete\n");
     
     return 0;
 }