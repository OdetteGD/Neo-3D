# Engine subsystem source

The Android Vulkan runtime currently draws three animated indexed cube meshes with per-frame transform matrices, depth testing, a GGX-style direct-light fragment shader, and touch-driven viewport orbit controls. Shader source under app/src/main/cpp/shaders is compiled to SPIR-V and embedded into the native library at build time.

- scene/: entity IDs, transforms, mesh references and PBR material data structures.
- render/: CPU screen-space light tile assignment, currently not wired to GPU shader buffers.
- assets/: GLB v2 container reader. It validates the binary container and extracts chunks; it is not yet a complete glTF semantic parser or GPU mesh uploader.

Remaining engine work includes connecting imported GLB assets to Vulkan vertex/index buffers, configurable material and light descriptors, shadow maps, image-based lighting, HDR post-processing, editor scene hierarchy/inspector, serialization, and on-device profiling. The native renderer must be validated by a passing Android build and actual device testing.
