# First-Person Camera Controller with Collision Detection

A complete OpenGL implementation of a first-person camera controller with robust collision detection and sliding mechanics.

## Features

- **First-Person Camera**
  - Mouse look with pitch/yaw control
  - WASD movement
  - Space/Shift for vertical movement
  - Mouse scroll for FOV zoom
  - Smooth movement with delta time

- **Collision Detection**
  - Axis-Aligned Bounding Box (AABB) collision
  - Sphere collision detection
  - Plane collision (floor/ceiling)
  - Camera represented as a capsule (radius + height)

- **Collision Response**
  - Slide along surfaces instead of stopping
  - Prevents clipping through walls
  - Smooth collision resolution

- **Demo Scene**
  - Room with walls
  - Box obstacles
  - Sphere obstacles
  - Phong lighting
  - Simple colored materials

## Controls

- **W/A/S/D** - Move forward/left/backward/right
- **Mouse** - Look around
- **Space** - Move up
- **Left Shift** - Move down
- **Mouse Scroll** - Zoom in/out
- **ESC** - Exit

## Dependencies

- **OpenGL 3.3+**
- **GLFW 3.x** - Window and input handling
- **GLM** - Mathematics library
- **GLAD** - OpenGL function loader

## Installation

### Ubuntu/Debian
```bash
sudo apt-get install libglfw3-dev libglm-dev cmake build-essential
```

### macOS (with Homebrew)
```bash
brew install glfw glm cmake
```

### Windows (with vcpkg)
```bash
vcpkg install glfw3 glm
```

### Download GLAD
1. Go to https://glad.dav1d.de/
2. Set:
   - Language: C/C++
   - API gl: Version 3.3+
   - Profile: Core
3. Generate and download
4. Extract `glad.c` and `include/` folder to project directory

## Building

```bash
mkdir build
cd build
cmake ..
make
./camera_demo
```

## Project Structure

```
.
├── Camera.h           # Camera class with movement and collision
├── Collision.h        # Collision detection manager
├── main.cpp          # Main application and rendering
├── CMakeLists.txt    # Build configuration
└── README.md         # This file
```

## Code Architecture

### Camera.h
- Camera class managing position, orientation, and movement
- Methods for keyboard and mouse input processing
- Collision check and resolution methods
- Sliding mechanics along collision surfaces

### Collision.h
- CollisionManager class for managing collision objects
- Support for AABB, Sphere, and Plane colliders
- Collision detection algorithms
- Normal calculation for collision response

### main.cpp
- OpenGL initialization and rendering loop
- Shader compilation and management
- Scene setup with collision objects
- Input processing with collision resolution
- Rendering of cubes and spheres

## Collision Detection Details

### Camera Representation
The camera is represented as a cylinder (capsule) with:
- **Radius**: 0.5 units (horizontal collision)
- **Height**: 1.8 units (vertical extent)

### Collision Types

1. **AABB Collision**
   - Checks if camera sphere intersects with box
   - Finds closest point on box to camera center
   - Calculates penetration and normal

2. **Sphere Collision**
   - Simple distance check between centers
   - Collision if distance < sum of radii

3. **Plane Collision**
   - Checks distance from camera to plane
   - Used for floor and ceiling

### Collision Response

When a collision is detected:
1. Store previous valid position
2. Calculate collision normal
3. Project movement vector onto surface (sliding)
4. If still colliding after slide, revert to previous position

This creates smooth wall-sliding behavior instead of getting stuck.

## Customization

### Adjust Camera Settings
In `Camera.h`, modify:
```cpp
const float SPEED       =  5.0f;   // Movement speed
const float SENSITIVITY =  0.1f;   // Mouse sensitivity
const float ZOOM        =  45.0f;  // Default FOV
```

### Add More Collision Objects
In `main.cpp`, add to collision manager:
```cpp
// Add a box
collisionManager.AddBox(glm::vec3(x1, y1, z1), glm::vec3(x2, y2, z2));

// Add a sphere
collisionManager.AddSphere(glm::vec3(x, y, z), radius);

// Add a plane
collisionManager.AddPlane(glm::vec3(nx, ny, nz), distance);
```

### Change Camera Size
In `Camera.h` constructor:
```cpp
CameraRadius(0.5f),   // Horizontal collision radius
CameraHeight(1.8f)    // Vertical height
```

## Performance Notes

- Collision detection is O(n) where n is number of collision objects
- For large scenes, consider spatial partitioning (octree, grid)
- Current implementation handles ~100 objects at 60+ FPS

## Future Enhancements

- [ ] Gravity and jumping mechanics
- [ ] Crouching (adjust camera height)
- [ ] Sprinting (increase movement speed)
- [ ] Spatial partitioning for large scenes
- [ ] Triangle mesh collision
- [ ] Stair climbing
- [ ] Slopes and ramps

## License

This code is provided as-is for educational and portfolio purposes.

## Credits

- GLM Mathematics Library
- GLFW for windowing
- GLAD for OpenGL loading



first-person-camera/
├── README.md                    # Comprehensive documentation
├── screenshots/                 # Visual demonstrations
│   ├── demo.gif
│   ├── collision_demo.png
│   └── scene_overview.png
├── src/
│   ├── Camera.h                 # Camera implementation
│   ├── Collision.h              # Collision system
│   └── main.cpp                 # Main application
├── docs/
│   ├── ARCHITECTURE.md          # System design
│   ├── ALGORITHMS.md            # Algorithm explanations
│   └── PERFORMANCE.md           # Performance analysis
└── CMakeLists.txt              # Build configuration