#include "global_header.h"

int main() {
    glfwInit();
    GLFWwindow* win =
        glfwCreateWindow(800, 600,
        "Persistent Mapping Demo", NULL, NULL);
    glfwMakeContextCurrent(win);
    glewInit();

    if (!glewIsSupported("GL_ARB_buffer_storage")) {
        printf("Persistent buffers not supported\n");
        return 0;
    }

    GLuint prog = makeProgram();
    glUseProgram(prog);

    createPersistentBuffer();

    while (!glfwWindowShouldClose(win)) {
        glClear(GL_COLOR_BUFFER_BIT);

        streamVertices();
        draw();

        glfwSwapBuffers(win);
        glfwPollEvents();
    }

    glfwTerminate();
    return 0;
}

/*
CPU profiler

GL_TIME_ELAPSED queries

Frame-time variance (stutter)


*/



/*
Method	CPU cost	Stutter
glBufferSubData	High	Yes
Persistent map	Near-zero	None

*/


/*
We are learning

How drivers manage memory residency

Why implicit sync kills performance

Why explicit fences matter

How engines stream millions of vertices/frame


*/