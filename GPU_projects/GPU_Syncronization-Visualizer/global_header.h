#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#define FRAMES 300
double cpuSubmit[FRAMES];
double cpuWait[FRAMES];
double gpuExec[FRAMES];

int mode = 0; // 0=none, 1=glFlush, 2=glFinish, 3=fence
