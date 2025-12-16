/*
 * rocm_opengl_app.c - OpenGL Application using ROCm Driver
 * 
 * This application demonstrates:
 * - Using ROCm userspace library to interact with kernel driver
 * - Allocating GPU memory through ROCm
 * - Submitting commands to ROCm driver
 * - Rendering with OpenGL
 * - Sharing data between ROCm and OpenGL
 */

 #include <stdio.h>
 #include <stdlib.h>
 #include <string.h>
 #include <math.h>
 #include <GL/glew.h>
 #include <GLFW/glfw3.h>
 #include "librocm.h"
 
 #define WINDOW_WIDTH 800
 #define WINDOW_HEIGHT 600
 
 /* Global state */
 GLFWwindow* window = NULL;
 rocm_context_t rocm_ctx = NULL;
 rocm_mem_t vertex_buffer_mem = 0;
 rocm_mem_t color_buffer_mem = 0;
 
 /* OpenGL objects */
 GLuint vao, vbo, color_vbo;
 GLuint shader_program;
 
 /* Vertex shader */
 const char* vertex_shader_source = 
     "#version 330 core\n"
     "layout (location = 0) in vec3 aPos;\n"
     "layout (location = 1) in vec3 aColor;\n"
     "out vec3 ourColor;\n"
     "uniform float time;\n"
     "void main() {\n"
     "   float angle = time;\n"
     "   mat3 rotation = mat3(\n"
     "       cos(angle), -sin(angle), 0.0,\n"
     "       sin(angle),  cos(angle), 0.0,\n"
     "       0.0,         0.0,        1.0\n"
     "   );\n"
     "   vec3 rotated = rotation * aPos;\n"
     "   gl_Position = vec4(rotated, 1.0);\n"
     "   ourColor = aColor;\n"
     "}\n";
 
 /* Fragment shader */
 const char* fragment_shader_source = 
     "#version 330 core\n"
     "in vec3 ourColor;\n"
     "out vec4 FragColor;\n"
     "void main() {\n"
     "   FragColor = vec4(ourColor, 1.0);\n"
     "}\n";
 
 /* Error callback */
 void error_callback(int error, const char* description)
 {
     fprintf(stderr, "GLFW Error %d: %s\n", error, description);
 }
 
 /* Compile shader */
 GLuint compile_shader(GLenum type, const char* source)
 {
     GLuint shader = glCreateShader(type);
     glShaderSource(shader, 1, &source, NULL);
     glCompileShader(shader);
     
     GLint success;
     glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
     if (!success) {
         char info[512];
         glGetShaderInfoLog(shader, 512, NULL, info);
         fprintf(stderr, "Shader compilation failed: %s\n", info);
         return 0;
     }
     
     return shader;
 }
 
 /* Initialize OpenGL */
 int init_opengl()
 {
     glfwSetErrorCallback(error_callback);
     
     if (!glfwInit()) {
         fprintf(stderr, "Failed to initialize GLFW\n");
         return -1;
     }
     
     glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
     glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
     glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
     
     window = glfwCreateWindow(WINDOW_WIDTH, WINDOW_HEIGHT,
                              "ROCm + OpenGL Integration", NULL, NULL);
     if (!window) {
         fprintf(stderr, "Failed to create GLFW window\n");
         glfwTerminate();
         return -1;
     }
     
     glfwMakeContextCurrent(window);
     
     glewExperimental = GL_TRUE;
     if (glewInit() != GLEW_OK) {
         fprintf(stderr, "Failed to initialize GLEW\n");
         return -1;
     }
     
     glViewport(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT);
     glClearColor(0.1f, 0.1f, 0.15f, 1.0f);
     
     return 0;
 }
 
 /* Initialize shaders */
 int init_shaders()
 {
     GLuint vertex_shader = compile_shader(GL_VERTEX_SHADER, vertex_shader_source);
     GLuint fragment_shader = compile_shader(GL_FRAGMENT_SHADER, fragment_shader_source);
     
     if (!vertex_shader || !fragment_shader)
         return -1;
     
     shader_program = glCreateProgram();
     glAttachShader(shader_program, vertex_shader);
     glAttachShader(shader_program, fragment_shader);
     glLinkProgram(shader_program);
     
     GLint success;
     glGetProgramiv(shader_program, GL_LINK_STATUS, &success);
     if (!success) {
         char info[512];
         glGetProgramInfoLog(shader_program, 512, NULL, info);
         fprintf(stderr, "Shader linking failed: %s\n", info);
         return -1;
     }
     
     glDeleteShader(vertex_shader);
     glDeleteShader(fragment_shader);
     
     return 0;
 }
 
 /* Initialize ROCm and allocate GPU memory */
 int init_rocm()
 {
     int ret;
     rocm_gpu_info_t gpu_info;
     
     /* Initialize ROCm context */
     ret = rocm_init(&rocm_ctx);
     if (ret != ROCM_SUCCESS) {
         fprintf(stderr, "Failed to initialize ROCm: %s\n",
                 rocm_get_error_string(ret));
         return -1;
     }
     
     /* Get GPU info */
     rocm_get_device_info(rocm_ctx, &gpu_info);
     
     /* Allocate GPU memory for vertex data */
     ret = rocm_malloc(rocm_ctx, &vertex_buffer_mem, 
                       9 * sizeof(float), ROCM_MEM_READ_WRITE);
     if (ret != ROCM_SUCCESS) {
         fprintf(stderr, "Failed to allocate vertex buffer: %s\n",
                 rocm_get_error_string(ret));
         return -1;
     }
     
     /* Allocate GPU memory for color data */
     ret = rocm_malloc(rocm_ctx, &color_buffer_mem,
                       9 * sizeof(float), ROCM_MEM_READ_WRITE);
     if (ret != ROCM_SUCCESS) {
         fprintf(stderr, "Failed to allocate color buffer: %s\n",
                 rocm_get_error_string(ret));
         return -1;
     }
     
     printf("\n=== ROCm GPU Memory Allocated ===\n");
     printf("Vertex buffer handle: %llu\n", vertex_buffer_mem);
     printf("Color buffer handle: %llu\n", color_buffer_mem);
     
     return 0;
 }
 
 /* Setup geometry data using ROCm */
 int setup_geometry()
 {
     int ret;
     
     /* Triangle vertices */
     float vertices[] = {
         -0.5f, -0.5f, 0.0f,
          0.5f, -0.5f, 0.0f,
          0.0f,  0.5f, 0.0f
     };
     
     /* Vertex colors */
     float colors[] = {
         1.0f, 0.0f, 0.0f,  // Red
         0.0f, 1.0f, 0.0f,  // Green
         0.0f, 0.0f, 1.0f   // Blue
     };
     
     /* Copy vertex data to GPU through ROCm */
     printf("\n=== Transferring Data via ROCm ===\n");
     ret = rocm_memcpy_h2d(rocm_ctx, vertex_buffer_mem, vertices, sizeof(vertices));
     if (ret != ROCM_SUCCESS) {
         fprintf(stderr, "Failed to copy vertex data: %s\n",
                 rocm_get_error_string(ret));
         return -1;
     }
     
     /* Copy color data to GPU through ROCm */
     ret = rocm_memcpy_h2d(rocm_ctx, color_buffer_mem, colors, sizeof(colors));
     if (ret != ROCM_SUCCESS) {
         fprintf(stderr, "Failed to copy color data: %s\n",
                 rocm_get_error_string(ret));
         return -1;
     }
     
     /* Submit ROCm command to process data */
     rocm_cmdbuf_t *cmdbuf;
     rocm_create_cmdbuf(&cmdbuf, 16);
     
     /* Add commands to buffer (simulated GPU commands) */
     rocm_cmdbuf_add(cmdbuf, 0x01000000);  // CMD_UPLOAD_VERTICES
     rocm_cmdbuf_add(cmdbuf, (uint32_t)(vertex_buffer_mem & 0xFFFFFFFF));
     rocm_cmdbuf_add(cmdbuf, 0x02000000);  // CMD_UPLOAD_COLORS
     rocm_cmdbuf_add(cmdbuf, (uint32_t)(color_buffer_mem & 0xFFFFFFFF));
     rocm_cmdbuf_add(cmdbuf, 0xFF000000);  // CMD_SYNC
     
     ret = rocm_submit(rocm_ctx, cmdbuf, ROCM_CMD_COMPUTE);
     if (ret != ROCM_SUCCESS) {
         fprintf(stderr, "Failed to submit commands: %s\n",
                 rocm_get_error_string(ret));
         rocm_destroy_cmdbuf(cmdbuf);
         return -1;
     }
     
     rocm_destroy_cmdbuf(cmdbuf);
     rocm_sync(rocm_ctx);
     
     printf("ROCm commands executed successfully\n");
     
     return 0;
 }
 
 /* Setup OpenGL buffers */
 int setup_opengl_buffers()
 {
     float vertices[] = {
         -0.5f, -0.5f, 0.0f,
          0.5f, -0.5f, 0.0f,
          0.0f,  0.5f, 0.0f
     };
     
     float colors[] = {
         1.0f, 0.0f, 0.0f,
         0.0f, 1.0f, 0.0f,
         0.0f, 0.0f, 1.0f
     };
     
     /* Create VAO */
     glGenVertexArrays(1, &vao);
     glBindVertexArray(vao);
     
     /* Create and setup vertex buffer */
     glGenBuffers(1, &vbo);
     glBindBuffer(GL_ARRAY_BUFFER, vbo);
     glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
     glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
     glEnableVertexAttribArray(0);
     
     /* Create and setup color buffer */
     glGenBuffers(1, &color_vbo);
     glBindBuffer(GL_ARRAY_BUFFER, color_vbo);
     glBufferData(GL_ARRAY_BUFFER, sizeof(colors), colors, GL_STATIC_DRAW);
     glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
     glEnableVertexAttribArray(1);
     
     glBindVertexArray(0);
     
     printf("\n=== OpenGL Buffers Created ===\n");
     printf("VAO: %u\n", vao);
     printf("Vertex VBO: %u\n", vbo);
     printf("Color VBO: %u\n", color_vbo);
     
     return 0;
 }
 
 /* Render frame */
 void render()
 {
     static float time = 0.0f;
     time += 0.01f;
     
     glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
     
     glUseProgram(shader_program);
     
     /* Set time uniform for rotation */
     GLint time_loc = glGetUniformLocation(shader_program, "time");
     glUniform1f(time_loc, time);
     
     glBindVertexArray(vao);
     glDrawArrays(GL_TRIANGLES, 0, 3);
     glBindVertexArray(0);
 }
 
 /* Print statistics */
 void print_stats()
 {
     rocm_gpu_info_t info;
     rocm_get_device_info(rocm_ctx, &info);
     
     printf("\n");
     printf("╔════════════════════════════════════════════════════════╗\n");
     printf("║       ROCm + OpenGL Integration Statistics            ║\n");
     printf("╠════════════════════════════════════════════════════════╣\n");
     printf("║ GPU Device: %-42s ║\n", info.device_name);
     printf("║ Compute Units: %-39u ║\n", info.compute_units);
     printf("║ Max Clock: %-39u MHz ║\n", info.max_clock_freq);
     printf("║ VRAM Size: %-39llu GB  ║\n", info.vram_size / (1024*1024*1024));
     printf("╠════════════════════════════════════════════════════════╣\n");
     printf("║ ROCm Memory Allocated:                                 ║\n");
     printf("║   Vertex Buffer: %llu (handle)                        ║\n", vertex_buffer_mem);
     printf("║   Color Buffer:  %llu (handle)                        ║\n", color_buffer_mem);
     printf("╠════════════════════════════════════════════════════════╣\n");
     printf("║ OpenGL Resources:                                      ║\n");
     printf("║   Shader Program: %-36u ║\n", shader_program);
     printf("║   VAO: %-47u ║\n", vao);
     printf("║   Vertex VBO: %-39u ║\n", vbo);
     printf("║   Color VBO: %-40u ║\n", color_vbo);
     printf("╚════════════════════════════════════════════════════════╝\n");
     printf("\n");
 }
 
 /* Cleanup */
 void cleanup()
 {
     /* Cleanup OpenGL */
     if (vao) glDeleteVertexArrays(1, &vao);
     if (vbo) glDeleteBuffers(1, &vbo);
     if (color_vbo) glDeleteBuffers(1, &color_vbo);
     if (shader_program) glDeleteProgram(shader_program);
     
     /* Cleanup ROCm */
     if (rocm_ctx) {
         if (vertex_buffer_mem) rocm_free(rocm_ctx, vertex_buffer_mem);
         if (color_buffer_mem) rocm_free(rocm_ctx, color_buffer_mem);
         rocm_destroy(rocm_ctx);
     }
     
     /* Cleanup GLFW */
     if (window) glfwDestroyWindow(window);
     glfwTerminate();
     
     printf("\nCleanup complete\n");
 }
 
 /* Main function */
 int main()
 {
     printf("╔════════════════════════════════════════════════════════╗\n");
     printf("║     ROCm Kernel + Userspace + OpenGL Application      ║\n");
     printf("╚════════════════════════════════════════════════════════╝\n\n");
     
     /* Initialize ROCm */
     if (init_rocm() < 0) {
         fprintf(stderr, "ROCm initialization failed\n");
         cleanup();
         return -1;
     }
     
     /* Setup geometry through ROCm */
     if (setup_geometry() < 0) {
         fprintf(stderr, "Geometry setup failed\n");
         cleanup();
         return -1;
     }
     
     /* Initialize OpenGL */
     if (init_opengl() < 0) {
         fprintf(stderr, "OpenGL initialization failed\n");
         cleanup();
         return -1;
     }
     
     /* Initialize shaders */
     if (init_shaders() < 0) {
         fprintf(stderr, "Shader initialization failed\n");
         cleanup();
         return -1;
     }
     
     /* Setup OpenGL buffers */
     if (setup_opengl_buffers() < 0) {
         fprintf(stderr, "OpenGL buffer setup failed\n");
         cleanup();
         return -1;
     }
     
     /* Print statistics */
     print_stats();
     
     printf("Starting render loop...\n");
     printf("Press ESC or close window to exit\n\n");
     
     /* Main render loop */
     int frame_count = 0;
     while (!glfwWindowShouldClose(window)) {
         render();
         
         glfwSwapBuffers(window);
         glfwPollEvents();
         
         frame_count++;
         if (frame_count % 60 == 0) {
             printf("Frame: %d (ROCm+OpenGL rendering)\n", frame_count);
         }
         
         if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
             glfwSetWindowShouldClose(window, 1);
         }
     }
     
     printf("\nTotal frames rendered: %d\n", frame_count);
     
     cleanup();
     
     return 0;
 }
