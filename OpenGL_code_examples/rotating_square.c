#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <math.h>
#include <stdio.h>

static const char *vs_src =
"#version 330 core\n"
"layout(location=0) in vec2 pos;\n"
"uniform float angle;\n"
"void main() {\n"
"    mat2 r = mat2(cos(angle), -sin(angle),\n"
"                  sin(angle),  cos(angle));\n"
"    gl_Position = vec4(r * pos, 0.0, 1.0);\n"
"}\n";

static const char *fs_src =
"#version 330 core\n"
"out vec4 color;\n"
"void main() { color = vec4(0.2, 0.7, 1.0, 1.0); }\n";

GLuint compile(GLenum type, const char *src)
{
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &src, NULL);
    glCompileShader(s);
    return s;
}

int main(void)
{
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow *w = glfwCreateWindow(600, 600, "Rotating Square", NULL, NULL);
    glfwMakeContextCurrent(w);
    glewInit();

    float verts[] = {
        -0.5f, -0.5f,
         0.5f, -0.5f,
         0.5f,  0.5f,
        -0.5f,  0.5f
    };

    GLuint idx[] = { 0,1,2, 2,3,0 };

    GLuint vao,vbo,ebo;
    glGenVertexArrays(1,&vao);
    glGenBuffers(1,&vbo);
    glGenBuffers(1,&ebo);

    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER,vbo);
    glBufferData(GL_ARRAY_BUFFER,sizeof(verts),verts,GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER,ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER,sizeof(idx),idx,GL_STATIC_DRAW);

    glVertexAttribPointer(0,2,GL_FLOAT,GL_FALSE,2*sizeof(float),(void*)0);
    glEnableVertexAttribArray(0);

    GLuint prog = glCreateProgram();
    glAttachShader(prog, compile(GL_VERTEX_SHADER,vs_src));
    glAttachShader(prog, compile(GL_FRAGMENT_SHADER,fs_src));
    glLinkProgram(prog);

    while (!glfwWindowShouldClose(w)) {
        float t = (float)glfwGetTime();
        glClear(GL_COLOR_BUFFER_BIT);

        glUseProgram(prog);
        glUniform1f(glGetUniformLocation(prog,"angle"), t);
        glBindVertexArray(vao);
        glDrawElements(GL_TRIANGLES,6,GL_UNSIGNED_INT,0);

        glfwSwapBuffers(w);
        glfwPollEvents();
    }

    glfwTerminate();
    return 0;
}
