# Ubuntu / Debian
#sudo apt install libglfw3-dev libglew-dev mesa-common-dev

gcc triangle.c -o triangle \
    -lglfw -lGLEW -lGL

./triangle
