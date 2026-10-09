# Neo-3D — Android Vulkan 3D Engine Foundation

Neo-3D is an Android-native C++20 Vulkan renderer project with an interactive touch viewport.

## Implemented source

- Android SurfaceView + JNI lifecycle integration.
- Native Vulkan instance, physical-device selection, Android presentation surface, swapchain, image views, graphics render pass, depth attachment, framebuffers, command buffers, fences and semaphores.
- Actual indexed 3D mesh draw calls: a 24-vertex / 36-index cube mesh, animated multi-object scene, vertex/index buffers and depth testing.
- Android document picker for `.glb` files, a dependency-free glTF JSON/accessor decoder for triangle primitives (`POSITION`, optional `NORMAL`, unsigned 8/16/32-bit indices), GPU vertex/index buffer upload, mesh bounds normalization, and indexed rendering of imported geometry.
- GLSL vertex and fragment shaders compiled to SPIR-V during the Android native build and embedded into the shared library.
- Metallic-roughness GGX-style direct-light shading with Fresnel and tone mapping.
- Touch drag orbit, auto-rotation toggle and reset-view controls.
- Scene/entity/transform/material data structures, a GLB v2 container reader, a dependency-free glTF JSON/accessor decoder for triangle primitives (POSITION, optional NORMAL, and unsigned indices), and CPU screen-tile light assignment code.

## Build

Requirements: JDK 17, Gradle 8.9, Android SDK platform 35, Android NDK 27.2.12479018, CMake 3.22.1, and glslangValidator (glslang-tools on Ubuntu).

Run:
```sh
gradle --no-daemon assembleDebug
gradle --no-daemon test
```

The GitHub Actions workflow installs the native toolchain, compiles GLSL into SPIR-V, builds the debug APK, runs tests, and uploads the APK artifact.

## Code destinations

- app/src/main/java/com/neo3d/engine/ — Android viewport/editor controls and JNI bridge.
- app/src/main/cpp/native_engine.cpp — Vulkan renderer and mesh draw loop.
- app/src/main/cpp/shaders/ — runtime mesh vertex/fragment shaders.
- app/src/main/cpp/tools/ — shader build-time utilities.
- engine/scene/ — scene entity, transform, mesh, and material data.
- engine/render/ — CPU-side light tile assignment foundation.
- engine/assets/ — GLB v2 container reader and asset import contract.

## Not yet complete

This is an actively developing engine foundation, not yet a production-complete mobile editor. The initial GLB geometry path now imports embedded-buffer triangle primitives and renders their positions/normals/indices. It does **not yet** support external `.gltf` buffers, sparse accessors, non-triangle primitives, UVs, textures, glTF material application, skinning, morph targets, or full scene/node transforms. Imported meshes currently use the renderer's hard-coded PBR shader/material. The current PBR shader uses a hard-coded material/light; the CPU tile assignment is not yet connected to GPU clustered/Forward+ shading. Shadow maps, image-based lighting, texture sampling, HDR off-screen rendering/bloom, a complete material/scene inspector, scene serialization, asset import UI, and performance/device testing remain to be integrated and verified. Treat APK build status as verified only after a successful GitHub Actions run.
