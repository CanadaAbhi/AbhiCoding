/*
 * global_header.h - Global definitions and includes
 * OpenGL Persistent Buffer Mapping Demo
 */

 #ifndef GLOBAL_HEADER_H
 #define GLOBAL_HEADER_H
 
 #include <stdio.h>
 #include <stdlib.h>
 #include <string.h>
 #include <math.h>
 
 #include <GL/glew.h>
 #include <GLFW/glfw3.h>
 
 /* Configuration */
 #define NUM_FRAMES 3              // Ring buffer (triple buffering)
 #define VERTS_PER_FRAME 10000     // Vertices per frame
 
 /* External shader sources */
 extern const char* vsSrc;
 extern const char* fsSrc;
 
 /* External buffer variables */
 extern GLuint vbo;
 extern void* mappedPtr;
 extern GLsync fences[NUM_FRAMES];
 extern int frameIndex;
 
 /* Function prototypes */
 
 /* Shader compilation */
 GLuint compile(GLenum type, const char* src);
 GLuint makeProgram(void);
 
 /* Buffer management */
 void createPersistentBuffer(void);
 void waitForFrame(int i);
 
 /* Data streaming */
 void streamVertices(void);
 void streamSubData(float* cpuData);
 
 /* Rendering */
 void draw(void);
 
 #endif /* GLOBAL_HEADER_H */