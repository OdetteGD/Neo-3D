# NEO-3D — Master Implementation Prompt

Repository: https://github.com/OdetteGD/Neo-3D

## Mission
Build a real, maintainable 3D game engine and editor, with native Vulkan GPU rendering, a genuine Forward+ pipeline, PBR materials, a functional scene editor, asset importing, Android packaging, and verified CI. Implement features in source code; do not stop at plans, mockups, or decorative UI. Never fabricate builds, tests, commits, artifacts, or rendering results.

## Mandatory repository audit
Inspect the actual branch, commit history, working tree, files, licenses, language, build system, dependencies, and existing CI before implementation. Read existing source before modifying it. Run baseline builds/tests where possible and record failures. Preserve user changes and working public APIs. If the repository is empty or source is inaccessible, state that fact and do not pretend features already exist.

## Renderer architecture
Implement a modular native Vulkan backend appropriate to the actual project and supported platforms:
- Vulkan instance/device selection, capability checks, graphics/presentation queues, swapchain lifecycle and recreation.
- Command pools/buffers, synchronization, frame scheduling, descriptor layouts/sets, pipeline layouts, shader management, vertex/index/uniform/storage buffers, texture uploads, depth, MSAA where supported, and GPU resource lifetime management.
- Validation-layer support in development, explicit initialization/device-loss errors, and diagnostics rather than silent blank viewports.
- Keep renderer APIs separate from editor, scene/runtime, asset pipeline, and platform integration. Use maintained dependencies and verify licenses before adopting them.

## Genuine Forward+ and graphics
Implement compute-shader tiled or clustered light culling with per-tile/cluster light lists and index buffers, correct synchronization/resource layouts, dynamic point/spot lights, directional lights, and configurable limits. Implement metallic-roughness PBR, GGX BRDF, Fresnel-Schlick, energy-conserving lighting, HDR, tone mapping, gamma-correct color handling, and GPU timing where supported.
Incrementally implement shadow maps (including cascades where practical), environment/image-based lighting, irradiance and prefiltered environment maps, normal/metallic/roughness/AO/emissive textures, mipmaps, anisotropic filtering when available, SSAO, bloom, exposure, color grading, optional SSR/DOF/motion blur, anti-aliasing, sky/environment controls, fog, terrain, and reflection probes. Every claimed effect must operate on real scene data and GPU resources; document unsupported or unimplemented features honestly.

## Functional editor and scene system
Implement an actual 3D viewport with orbit/pan/zoom/fly camera, selection, transform gizmos, local/global modes, hierarchy, inspector, asset browser, material and lighting controls, renderer settings, console/error panel, play/pause/stop/restart, undo/redo, dirty state, and save/load. Controls must be connected to real engine behavior.
Use stable scene/object identifiers, transform hierarchies, camera/mesh/light components, lifecycle/update logic, versioned serialization, resource references, and clear missing-resource errors.

## Asset pipeline
Prioritize glTF 2.0 and GLB. Import actual mesh geometry, supported materials, textures, and animations. Implement texture decoding/GPU upload, source preservation, generated caches, invalidation/reload, dependency tracking, and readable import errors. Include a real sample scene with geometry, materials, and lighting.

## Android APK and platform separation
Inspect the actual app architecture, Android entry point, Gradle/AGP/JDK/SDK/NDK/CMake requirements, native libraries, shader pipeline, and renderer before implementing Android support. Add `.github/workflows/android-apk.yml` that builds a real installable APK using the repository's actual build system, runs applicable tests and static checks, validates the output, and uploads it as a GitHub Actions artifact. Support manual dispatch and relevant pushes/PRs. Use least-privilege permissions, cache safely, pin third-party actions to verified full SHAs where practical, and never expose secrets.
Do not invent an Android module or build command without verifying the project. If source/build files are missing, fail with a clear diagnostic rather than manufacturing an APK. Keep Android platform code separate from desktop windowing and desktop renderer behavior. Do not switch renderers or remove shaders/features merely to make Android compilation pass. Validate Android Vulkan feature/device requirements, lifecycle, native surface/swapchain resize and recreation, pause/resume, input, shader/resource packaging, and unsupported-feature fallbacks. A successful APK compilation is not proof that GPU rendering works on a device.

## CI, tests, and regression protection
Create CI checks appropriate to the actual codebase:
- Android compile/tests/lint/package validation and artifact upload.
- Desktop engine/editor builds and tests.
- Renderer and shader compilation/validation.
- Scene serialization, math, resource management, GLB import, and material loading tests where implemented.
- Workflow syntax/configuration checks.
Do not claim a test passed if skipped or unable to run. Distinguish compile checks, unit tests, emulator/device tests, and GPU rendering tests. If physical hardware is unavailable, disclose that limitation.
After each relevant push, wait for all required workflows to finish. Inspect failing job steps and logs, find root causes, make the smallest safe fix, commit and push it, and repeat. Do not disable checks, delete tests, or weaken validation to obtain green status. Do not stop while required checks are running. If an external blocker prevents completion, report the exact blocker and next action.

## Milestones (implementation gates, not a substitute for implementation)
1. Audit repository and establish verified baseline.
2. Build a real Vulkan viewport and minimal scene.
3. Add PBR and genuine Forward+ compute light culling.
4. Add shadows, HDR, tone mapping, and further effects incrementally.
5. Build functional editor and scene serialization.
6. Add GLB/glTF import, material editing, diagnostics, profiling.
7. Add Android APK workflow, automated tests, documentation, packaging.
At each gate: change real source, build affected targets, run relevant tests, fix regressions, document actual results, and commit a coherent working change. Do not claim the full engine is complete merely because the workflow or documentation exists.

## Git and release safety
Inspect current Git status/history before editing. Use a dedicated feature branch when an existing base commit permits it. Preserve unrelated changes. Commit coherent milestones with descriptive messages and push only when authenticated write access permits. Never force-push, rewrite shared history, commit secrets, or claim a commit/push unless confirmed. Review repository and third-party licenses and preserve required notices.

## Definition of done
The usable release must build on documented supported platforms, launch a real editor, render actual GPU geometry through Vulkan, expose working viewport camera controls, use PBR and a genuine Forward+ path, render shadows where supported, import a GLB asset, save/reload scenes, and provide functioning hierarchy/inspector/transform tools. Android must produce a validated APK artifact and preserve desktop build/rendering checks. Report exact commit hashes, workflow runs, artifacts, test results, and limitations. Never fabricate success.

## Final execution instruction
Start with the actual repository state. Implement features directly in source; do not turn the milestones into another planning-only deliverable. Add and validate the Android APK workflow without compromising rendering. Continue through failures and rerun checks until all required checks pass, or clearly identify a genuine blocker that cannot be resolved with available source, permissions, hardware, or infrastructure.
