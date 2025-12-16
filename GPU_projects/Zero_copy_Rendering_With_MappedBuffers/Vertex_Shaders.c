/*
 * vertex_shaders.c - Shader source definitions
 */

 #include "global_header.h"

 /* Vertex shader source */
 const char* vsSrc =
     "#version 440 core\n"
     "layout(location=0) in vec2 pos;\n"
     "void main() {\n"
     "    gl_Position = vec4(pos, 0.0, 1.0);\n"
     "}\n";
 
 /* Fragment shader source */
 const char* fsSrc =
     "#version 440 core\n"
     "out vec4 FragColor;\n"
     "void main() {\n"
     "    FragColor = vec4(1.0, 1.0, 1.0, 1.0);\n"
     "}\n";