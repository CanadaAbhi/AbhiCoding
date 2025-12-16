#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <math.h>
#include <stdio.h>

static const char *vs_src =
"#version 330 core\n"
"layout(location=0) in vec3 pos;\n"
"uniform mat4 mvp;\n"
"void main() { gl_Position = mvp * vec4(pos,1.0); }\n";

static const char *fs_src =
"#version 330 core\n"
"out vec4 color;\n"
"void main() { color = vec4(1.0,0.4,0.2,1.0); }\n";

GLuint compile(GLenum t,const char* s){
    GLuint sh=glCreateShader(t);
    glShaderSource(sh,1,&s,NULL);
    glCompileShader(sh);
    return sh;
}

/* Simple perspective * rotation matrix */
void mat4(float *m, float a)
{
    float c=cos(a), s=sin(a);
    m[0]= c; m[4]=0; m[8]= s; m[12]=0;
    m[1]= 0; m[5]=1; m[9]= 0; m[13]=0;
    m[2]=-s; m[6]=0; m[10]=c; m[14]=-3;
    m[3]= 0; m[7]=0; m[11]=0; m[15]=1;
}

int main(void)
{
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR,3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR,3);
    glfwWindowHint(GLFW_OPENGL_PROFILE,GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow *w=glfwCreateWindow(800,600,"Rotating Cube",NULL,NULL);
    glfwMakeContextCurrent(w);
    glewInit();

    glEnable(GL_DEPTH_TEST);

    float v[]={
        -1,-1,-1,  1,-1,-1,  1, 1,-1, -1, 1,-1,
        -1,-1, 1,  1,-1, 1,  1, 1, 1, -1, 1, 1
    };

    GLuint idx[]={
        0,1,2,2,3,0, 4,5,6,6,7,4,
        0,4,7,7,3,0, 1,5,6,6,2,1,
        3,2,6,6,7,3, 0,1,5,5,4,0
    };

    GLuint vao,vbo,ebo;
    glGenVertexArrays(1,&vao);
    glGenBuffers(1,&vbo);
    glGenBuffers(1,&ebo);

    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER,vbo);
    glBufferData(GL_ARRAY_BUFFER,sizeof(v),v,GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER,ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER,sizeof(idx),idx,GL_STATIC_DRAW);

    glVertexAttribPointer(0,3,GL_FLOAT,GL_FALSE,3*sizeof(float),(void*)0);
    glEnableVertexAttribArray(0);

    GLuint prog=glCreateProgram();
    glAttachShader(prog,compile(GL_VERTEX_SHADER,vs_src));
    glAttachShader(prog,compile(GL_FRAGMENT_SHADER,fs_src));
    glLinkProgram(prog);

    while(!glfwWindowShouldClose(w)){
        float mvp[16];
        mat4(mvp,(float)glfwGetTime());

        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);
        glUseProgram(prog);
        glUniformMatrix4fv(glGetUniformLocation(prog,"mvp"),
                           1,GL_FALSE,mvp);
        glBindVertexArray(vao);
        glDrawElements(GL_TRIANGLES,36,GL_UNSIGNED_INT,0);

        glfwSwapBuffers(w);
        glfwPollEvents();
    }
    glfwTerminate();
    return 0;
}
