# Neo-3D

Android-native C++ Vulkan engine project.

## Build requirements

- JDK 17
- Gradle 8.9
- Android SDK platform 35
- Android NDK 27.2.12479018
- CMake 3.22.1

Build with `gradle assembleDebug`; run tests with `gradle test`. GitHub Actions installs the Android native toolchain, builds a debug APK, runs tests, and uploads the APK artifact.

## Source destinations

- `app/src/main/java/com/neo3d/engine/`: Android lifecycle, viewport surface, and JNI bridge.
- `app/src/main/cpp/`: C++ Vulkan runtime and native build target.
- `engine/scene/`: entity, transform, mesh, and material data structures.
- `engine/shaders/`: GLSL forward vertex and GGX metallic-roughness fragment shaders.
- `engine/assets/`: imported model, texture, material, and scene asset destinations.

## Current renderer boundary

The integrated Vulkan viewport currently creates a native Vulkan surface, selects a presentation-capable GPU, creates a swapchain and render pass, records command buffers, synchronizes frames, and presents cleared swapchain images. The PBR shader files and scene data structures are source foundations and are not yet wired to GPU mesh rendering. Clustered lighting, shadows, HDR/post effects, glTF/GLB GPU upload, and a full editor remain unfinished. Treat the APK as build-verified only after the GitHub Actions run passes.
