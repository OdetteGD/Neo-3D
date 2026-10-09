# Neo-3D — Android Vulkan 3D Engine

Neo-3D is an Android-native C++20 Vulkan renderer with a touch-operated viewport and an in-progress GLB geometry import path.

## Active Vulkan frame pipeline

1. Acquire a swapchain image and wait on the per-frame fence.
2. Begin the swapchain render pass with color and depth attachments.
3. Draw the procedural sky fullscreen triangle.
4. Bind the indexed mesh pipeline and upload the model/view-projection push constants.
5. Run the mesh fragment shader: Cook–Torrance microfacet BRDF (GGX distribution, Smith/Schlick geometry, Fresnel), three directional-light contributions, a hemispherical ambient approximation, ACES-style tone mapping, and display gamma conversion.
6. Submit the command buffer, signal the render-finished semaphore, and present the swapchain image.

This is a real Vulkan render loop and shader pipeline, but it is not yet a full modern production renderer. The current material values and lights are shader constants; the ambient term is an approximation, not image-based lighting. There is no shadow-map pass or HDR post-processing pass yet.

## Implemented source

- Android SurfaceView + JNI lifecycle integration.
- Vulkan instance/device selection, Android presentation surface, swapchain, image views, color/depth render pass, framebuffers, command buffers, fences, semaphores, and resize rebuild path.
- Indexed 3D mesh draw calls, depth testing, an animated multi-object default scene, and GPU vertex/index buffers.
- Android document picker for .glb files, dependency-free glTF JSON/accessor decoding for triangle primitives (POSITION, optional NORMAL, unsigned 8/16/32-bit indices), bounds normalization, GPU upload, and indexed rendering.
- GLSL vertex/fragment shaders compiled to SPIR-V during the Android native build and embedded into the shared library.
- GGX/Cook–Torrance direct lighting, multiple fixed directional lights, Fresnel, hemispherical ambient approximation, ACES-style tone mapping, and procedural sky.
- Touch-drag orbit, auto-rotation toggle, and reset-view controls.
- Basic scene/entity/transform/material data structures and CPU screen-tile light assignment foundation.

## Build

Requirements: JDK 17, Gradle 8.9, Android SDK platform 35, Android NDK 27.2.12479018, CMake 3.22.1, and glslangValidator (glslang-tools on Ubuntu).

Run:
```sh
gradle --no-daemon assembleDebug
gradle --no-daemon testDebugUnitTest
```

GitHub Actions uses the Android SDK already present on the hosted Ubuntu runner and installs the required platform, build-tools, NDK, CMake, and shader compiler packages. This deliberately avoids the obsolete SDK Manager package named tools, which caused the previous workflow to fail before the native build started. The APK and test status are considered verified only after a successful Actions run.

## Code destinations

- app/src/main/java/com/neo3d/engine/ — Android viewport/import controls and JNI bridge.
- app/src/main/cpp/native_engine.cpp — Vulkan device/swapchain lifecycle, render loop, and mesh draw calls.
- app/src/main/cpp/shaders/ — runtime mesh and sky shaders.
- app/src/main/cpp/CMakeLists.txt — shader-to-SPIR-V compilation and embedding.
- engine/scene/ — scene entity, transform, mesh, and material data.
- engine/render/ — CPU-side light-tile assignment foundation.
- engine/assets/ — GLB v2 container reader and glTF geometry decoder.

## Known renderer gaps (not claimed as implemented)

The current GLB path does not yet support external .gltf buffers, sparse accessors, UVs, textures, applying glTF materials, skinning, morph targets, or node/scene transforms. All imported meshes still use the shader's default hard-coded material. CPU tile assignment is not connected to GPU Forward+/clustered light lists. Shadow maps, image-based lighting (environment cubemaps/prefiltered BRDF), GPU-driven/clustered lighting, HDR off-screen targets and bloom, a scene hierarchy/material inspector, scene serialization, broader asset import, and physical-device/performance testing remain unfinished. These are separate renderer features and must not be inferred from the existing Vulkan draw loop.
