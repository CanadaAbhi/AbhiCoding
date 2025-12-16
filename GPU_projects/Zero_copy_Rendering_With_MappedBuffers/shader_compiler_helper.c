/*
 * shader_compiler_helper.c - Shader compilation and buffer management
 */

 #include "global_header.h"

 /* Global buffer variables */
 GLuint vbo = 0;
 void* mappedPtr = NULL;
 GLsync fences[NUM_FRAMES] = {0};
 int frameIndex = 0;
 
 /**
  * compile - Compile a shader from source
  * @type: Shader type (GL_VERTEX_SHADER or GL_FRAGMENT_SHADER)
  * @src: Shader source code string
  *
  * Returns: Compiled shader ID
  */
 GLuint compile(GLenum type, const char* src)
 {
     GLuint shader;
     GLint success;
     GLchar infoLog[512];
 
     shader = glCreateShader(type);
     if (shader == 0) {
         fprintf(stderr, "Error: Failed to create shader\n");
         return 0;
     }
 
     glShaderSource(shader, 1, &src, NULL);
     glCompileShader(shader);
 
     /* Check compilation status */
     glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
     if (!success) {
         glGetShaderInfoLog(shader, 512, NULL, infoLog);
         fprintf(stderr, "Shader compilation failed:\n%s\n", infoLog);
         glDeleteShader(shader);
         return 0;
     }
 
     return shader;
 }
 
 /**
  * makeProgram - Create and link shader program
  *
  * Returns: Linked program ID
  */
 GLuint makeProgram(void)
 {
     GLuint program, vs, fs;
     GLint success;
     GLchar infoLog[512];
 
     /* Compile shaders */
     vs = compile(GL_VERTEX_SHADER, vsSrc);
     if (vs == 0)
         return 0;
 
     fs = compile(GL_FRAGMENT_SHADER, fsSrc);
     if (fs == 0) {
         glDeleteShader(vs);
         return 0;
     }
 
     /* Create program */
     program = glCreateProgram();
     if (program == 0) {
         fprintf(stderr, "Error: Failed to create program\n");
         glDeleteShader(vs);
         glDeleteShader(fs);
         return 0;
     }
 
     /* Attach shaders and link */
     glAttachShader(program, vs);
     glAttachShader(program, fs);
     glLinkProgram(program);
 
     /* Check link status */
     glGetProgramiv(program, GL_LINK_STATUS, &success);
     if (!success) {
         glGetProgramInfoLog(program, 512, NULL, infoLog);
         fprintf(stderr, "Program linking failed:\n%s\n", infoLog);
         glDeleteProgram(program);
         glDeleteShader(vs);
         glDeleteShader(fs);
         return 0;
     }
 
     /* Clean up shaders (no longer needed after linking) */
     glDeleteShader(vs);
     glDeleteShader(fs);
 
     return program;
 }
 
 /**
  * createPersistentBuffer - Set up persistent mapped buffer
  *
  * Creates a triple-buffered VBO with persistent mapping for zero-copy streaming
  */
 void createPersistentBuffer(void)
 {
     size_t totalSize;
 
     totalSize = NUM_FRAMES * VERTS_PER_FRAME * sizeof(float) * 2;
 
     glGenBuffers(1, &vbo);
     glBindBuffer(GL_ARRAY_BUFFER, vbo);
 
     /* Create buffer storage with persistent mapping flags */
     glBufferStorage(GL_ARRAY_BUFFER,
                     totalSize,
                     NULL,
                     GL_MAP_WRITE_BIT |
                     GL_MAP_PERSISTENT_BIT |
                     GL_MAP_COHERENT_BIT);
 
     /* Map the buffer persistently */
     mappedPtr = glMapBufferRange(GL_ARRAY_BUFFER,
                                   0,
                                   totalSize,
                                   GL_MAP_WRITE_BIT |
                                   GL_MAP_PERSISTENT_BIT |
                                   GL_MAP_COHERENT_BIT);
 
     if (mappedPtr == NULL) {
         fprintf(stderr, "Error: Failed to map buffer\n");
         exit(EXIT_FAILURE);
     }
 
     printf("Persistent buffer mapped @ %p (size: %zu bytes)\n", 
            mappedPtr, totalSize);
 }
 
 /**
  * waitForFrame - Wait for GPU to finish with frame buffer
  * @i: Frame index to wait for
  *
  * Uses GL sync objects to prevent CPU from overwriting data the GPU is using
  */
 void waitForFrame(int i)
 {
     if (fences[i] != 0) {
         GLenum result;
 
         /* Wait for fence to be signaled */
         while (1) {
             result = glClientWaitSync(fences[i],
                                       GL_SYNC_FLUSH_COMMANDS_BIT,
                                       1000000); /* 1ms timeout */
 
             if (result == GL_ALREADY_SIGNALED ||
                 result == GL_CONDITION_SATISFIED) {
                 break;
             }
 
             if (result == GL_WAIT_FAILED) {
                 fprintf(stderr, "Warning: Sync wait failed\n");
                 break;
             }
         }
 
         /* Delete the sync object */
         glDeleteSync(fences[i]);
         fences[i] = 0;
     }
 }
 
 /**
  * streamVertices - Generate and stream vertex data (zero-copy)
  *
  * Writes directly to mapped GPU memory, generating a sine wave
  */
 void streamVertices(void)
 {
     float* ptr;
     int i;
 
     /* Wait for this frame slot to be free */
     waitForFrame(frameIndex);
 
     /* Calculate pointer to current frame's data */
     ptr = (float*)mappedPtr + (frameIndex * VERTS_PER_FRAME * 2);
 
     /* Generate vertex data (sine wave) */
     for (i = 0; i < VERTS_PER_FRAME; i++) {
         float x = (float)i / (float)VERTS_PER_FRAME * 2.0f - 1.0f;
         float t = (float)glfwGetTime();
         
         ptr[i * 2]     = x;
         ptr[i * 2 + 1] = 0.5f * sinf(x * 10.0f + t * 2.0f);
     }
 }
 
 /**
  * streamSubData - Alternative: Copy data using glBufferSubData
  * @cpuData: Source data to copy
  *
  * This is the old method - slower due to CPU-GPU copy overhead
  */
 void streamSubData(float* cpuData)
 {
     glBindBuffer(GL_ARRAY_BUFFER, vbo);
     glBufferSubData(GL_ARRAY_BUFFER, 0,
                     VERTS_PER_FRAME * sizeof(float) * 2,
                     cpuData);
 }