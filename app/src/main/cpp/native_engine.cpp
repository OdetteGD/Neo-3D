#include <jni.h>
#include <android/native_window.h>
#include <android/native_window_jni.h>
#include <android/log.h>
#define VK_USE_PLATFORM_ANDROID_KHR
#include <vulkan/vulkan.h>
#include "shader_blobs.hpp"
#include "assets/glb_reader.hpp"
#include "assets/gltf_mesh_reader.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, "Neo3D", __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, "Neo3D", __VA_ARGS__)

namespace neo3d {

static void check(VkResult r, const char* op) {
    if (r != VK_SUCCESS) {
        LOGE("%s failed: VkResult=%d", op, static_cast<int>(r));
        throw std::runtime_error(op);
    }
}

// -----------------------------------------------------------------------------
// Vector at Matrix Math Engine (Expanded para sa Shadows at Tangents)
// -----------------------------------------------------------------------------
struct Vec2 {
    float x{0.0f}, y{0.0f};
};

struct Vec3 {
    float x{0.0f}, y{0.0f}, z{0.0f};

    Vec3 operator+(const Vec3& o) const { return {x + o.x, y + o.y, z + o.z}; }
    Vec3 operator-(const Vec3& o) const { return {x - o.x, y - o.y, z - o.z}; }
    Vec3 operator*(float s) const { return {x * s, y * s, z * s}; }
    Vec3& operator+=(const Vec3& o) { x += o.x; y += o.y; z += o.z; return *this; }
    float length() const { return std::sqrt(x * x + y * y + z * z); }
    Vec3 normalized() const {
        float l = length();
        return l > 1e-6f ? Vec3{x / l, y / l, z / l} : Vec3{0, 0, 0};
    }
    static Vec3 cross(const Vec3& a, const Vec3& b) {
        return {
            a.y * b.z - a.z * b.y,
            a.z * b.x - a.x * b.z,
            a.x * b.y - a.y * b.x
        };
    }
    static float dot(const Vec3& a, const Vec3& b) {
        return a.x * b.x + a.y * b.y + a.z * b.z;
    }
};

struct Vec4 {
    float x{0.0f}, y{0.0f}, z{0.0f}, w{0.0f};
};

struct Mat4 {
    float v[16]{};
};

static Mat4 identity() {
    Mat4 m{};
    m.v[0] = m.v[5] = m.v[10] = m.v[15] = 1.0f;
    return m;
}

static Mat4 multiply(const Mat4& a, const Mat4& b) {
    Mat4 o{};
    for (int c = 0; c < 4; ++c) {
        for (int r = 0; r < 4; ++r) {
            for (int k = 0; k < 4; ++k) {
                o.v[c * 4 + r] += a.v[k * 4 + r] * b.v[c * 4 + k];
            }
        }
    }
    return o;
}

static Mat4 perspective(float aspect) {
    const float f = 1.0f / std::tan(0.78539816339f * 0.5f); // 45 degrees FOV
    const float nearP = 0.1f, farP = 500.0f;
    Mat4 m{};
    m.v[0] = f / std::max(aspect, 0.01f);
    m.v[5] = -f; // Inverted Y para sa Vulkan Clip Space
    m.v[10] = farP / (nearP - farP);
    m.v[11] = -1.0f;
    m.v[14] = (farP * nearP) / (nearP - farP);
    return m;
}

static Mat4 ortho(float left, float right, float bottom, float top, float nearP, float farP) {
    Mat4 m = identity();
    m.v[0] = 2.0f / (right - left);
    m.v[5] = -2.0f / (top - bottom); // Inverted Y para sa Vulkan
    m.v[10] = 1.0f / (nearP - farP);
    m.v[12] = -(right + left) / (right - left);
    m.v[13] = -(top + bottom) / (top - bottom);
    m.v[14] = nearP / (nearP - farP);
    return m;
}

static Mat4 translate(float x, float y, float z) {
    Mat4 m = identity();
    m.v[12] = x; m.v[13] = y; m.v[14] = z;
    return m;
}

static Mat4 scale(float sx, float sy, float sz) {
    Mat4 m = identity();
    m.v[0] = sx; m.v[5] = sy; m.v[10] = sz;
    return m;
}

static Mat4 lookAt(const Vec3& eye, const Vec3& target, const Vec3& up) {
    Vec3 f = (target - eye).normalized();
    Vec3 s = Vec3::cross(f, up).normalized();
    Vec3 u = Vec3::cross(s, f);

    Mat4 m = identity();
    m.v[0] = s.x;  m.v[1] = u.x;  m.v[2] = -f.x;
    m.v[4] = s.y;  m.v[5] = u.y;  m.v[6] = -f.y;
    m.v[8] = s.z;  m.v[9] = u.z;  m.v[10] = -f.z;
    m.v[12] = -(s.x * eye.x + s.y * eye.y + s.z * eye.z);
    m.v[13] = -(u.x * eye.x + u.y * eye.y + u.z * eye.z);
    m.v[14] =  (f.x * eye.x + f.y * eye.y + f.z * eye.z);
    return m;
}

static Mat4 inverseMat4(const Mat4& m) {
    Mat4 inv{};
    const float* a = m.v;
    float b00 = a[0]*a[5] - a[1]*a[4], b01 = a[0]*a[6] - a[2]*a[4], b02 = a[0]*a[7] - a[3]*a[4];
    float b03 = a[1]*a[6] - a[2]*a[5], b04 = a[1]*a[7] - a[3]*a[5], b05 = a[2]*a[7] - a[3]*a[6];
    float b06 = a[8]*a[13]- a[9]*a[12],b07 = a[8]*a[14]- a[10]*a[12],b08 = a[8]*a[15]- a[11]*a[12];
    float b09 = a[9]*a[14]- a[10]*a[13],b10 = a[9]*a[15]- a[11]*a[13],b11 = a[10]*a[15]- a[11]*a[14];
    float det = b00*b11 - b01*b10 + b02*b09 + b03*b08 - b04*b07 + b05*b06;
    if (std::abs(det) < 1e-8f) return identity();
    float invDet = 1.0f / det;
    inv.v[0] = (a[5]*b11 - a[6]*b10 + a[7]*b09) * invDet;
    inv.v[1] = (-a[1]*b11 + a[2]*b10 - a[3]*b09) * invDet;
    inv.v[2] = (a[13]*b05 - a[14]*b04 + a[15]*b03) * invDet;
    inv.v[3] = (-a[9]*b05 + a[10]*b04 - a[11]*b03) * invDet;
    inv.v[4] = (-a[4]*b11 + a[6]*b08 - a[7]*b07) * invDet;
    inv.v[5] = (a[0]*b11 - a[2]*b08 + a[3]*b07) * invDet;
    inv.v[6] = (-a[12]*b05 + a[14]*b02 - a[15]*b01) * invDet;
    inv.v[7] = (a[8]*b05 - a[10]*b02 + a[11]*b01) * invDet;
    inv.v[8] = (a[4]*b10 - a[5]*b08 + a[7]*b06) * invDet;
    inv.v[9] = (-a[0]*b10 + a[1]*b08 - a[3]*b06) * invDet;
    inv.v[10] = (a[12]*b04 - a[13]*b02 + a[15]*b00) * invDet;
    inv.v[11] = (-a[8]*b04 + a[9]*b02 - a[11]*b00) * invDet;
    inv.v[12] = (-a[4]*b09 + a[5]*b07 - a[6]*b06) * invDet;
    inv.v[13] = (a[0]*b09 - a[1]*b07 + a[2]*b06) * invDet;
    inv.v[14] = (-a[12]*b03 + a[13]*b01 - a[14]*b00) * invDet;
    inv.v[15] = (a[8]*b03 - a[9]*b01 + a[10]*b00) * invDet;
    return inv;
}

// -----------------------------------------------------------------------------
// High-Fidelity PBR Vertex Format
// -----------------------------------------------------------------------------
struct Vertex {
    float position[3];
    float normal[3];
    float uv[2];
    float tangent[4]; // w holds handedness (1.0 o -1.0) para sa normal mapping
};

struct AABB {
    Vec3 min;
    Vec3 max;

    bool intersects(const AABB& o) const {
        return (min.x <= o.max.x && max.x >= o.min.x) &&
               (min.y <= o.max.y && max.y >= o.min.y) &&
               (min.z <= o.max.z && max.z >= o.min.z);
    }
};

// -----------------------------------------------------------------------------
// Global Scene UBO (UE4 / Unity Forward+ Standard)
// -----------------------------------------------------------------------------
struct alignas(16) SceneUniformBuffer {
    Mat4 viewProj;
    Mat4 invViewProj;
    Mat4 shadowViewProj;

    Vec4 cameraPosition;      // xyz = eye pos, w = time
    Vec4 sunDirection;       // xyz = normalized light vector, w = sun intensity
    Vec4 sunColor;           // rgb = direct light color, w = ambient intensity

    Vec4 fogAtmosphereParams;// x = density, y = timeOfDay, z = Rayleigh coef, w = Mie coef
    Vec4 cloudParameters;    // x = cloudCoverage, y = cloudSpeed, z = altitude, w = density
    Vec4 renderingSettings;  // x = exposure, y = shadowBias, z = pcfRadius, w = unused
};

struct ObjectPushConstants {
    Mat4 model;
    float baseColor[4];
    float pbrParams[4];      // x = metallic, y = roughness, z = ao, w = isWater
    float emissive[4];
};

struct SkyPushConstants {
    Mat4 invViewProj;
    float cameraPos[4];
    float sunDir[4];
    float envParams[4];
};

struct PhysicsObstacle {
    AABB box;
    bool isWalkable{false};
};

class CharacterController {
public:
    Vec3 position{0.0f, 2.0f, 8.0f};
    Vec3 velocity{0.0f, 0.0f, 0.0f};
    bool isGrounded{false};
    float eyeHeight{1.72f};
    float radius{0.35f};
    float height{1.80f};

    AABB getAABB(const Vec3& pos) const {
        return {
            {pos.x - radius, pos.y, pos.z - radius},
            {pos.x + radius, pos.y + height, pos.z + radius}
        };
    }

    void jump() {
        if (isGrounded) {
            velocity.y = 8.5f;
            isGrounded = false;
        }
    }

    void update(float dt, float inputX, float inputY, float yaw, const std::vector<PhysicsObstacle>& obstacles) {
        float sinY = std::sin(yaw);
        float cosY = std::cos(yaw);
        Vec3 forward = {sinY, 0.0f, -cosY};
        Vec3 right = {cosY, 0.0f, sinY};

        float speed = 8.0f;
        Vec3 targetMove = (forward * inputY + right * inputX) * speed;

        velocity.x = targetMove.x;
        velocity.z = targetMove.z;

        const float kGravity = -22.0f;
        velocity.y += kGravity * dt;
        if (velocity.y < -35.0f) velocity.y = -35.0f;

        const float stepOffset = 0.35f;

        // X Movement Check
        Vec3 stepX = position;
        stepX.x += velocity.x * dt;
        AABB boxX = getAABB(stepX);
        boxX.min.y += stepOffset;

        bool colX = false;
        for (const auto& obs : obstacles) {
            if (boxX.intersects(obs.box)) { colX = true; break; }
        }
        if (!colX) position.x = stepX.x;

        // Z Movement Check
        Vec3 stepZ = position;
        stepZ.z += velocity.z * dt;
        AABB boxZ = getAABB(stepZ);
        boxZ.min.y += stepOffset;

        bool colZ = false;
        for (const auto& obs : obstacles) {
            if (boxZ.intersects(obs.box)) { colZ = true; break; }
        }
        if (!colZ) position.z = stepZ.z;

        // Y (Vertical) Movement Check
        Vec3 stepY = position;
        stepY.y += velocity.y * dt;
        AABB boxY = getAABB(stepY);

        isGrounded = false;
        if (stepY.y <= 0.0f) {
            position.y = 0.0f;
            velocity.y = 0.0f;
            isGrounded = true;
        } else {
            bool colY = false;
            for (const auto& obs : obstacles) {
                if (boxY.intersects(obs.box)) {
                    if (velocity.y < 0.0f) {
                        position.y = obs.box.max.y;
                        velocity.y = 0.0f;
                        isGrounded = true;
                    } else if (velocity.y > 0.0f) {
                        position.y = obs.box.min.y - height;
                        velocity.y = 0.0f;
                    }
                    colY = true;
                    break;
                }
            }
            if (!colY) position.y = stepY.y;
        }
    }

    Vec3 getEyePosition() const {
        return {position.x, position.y + eyeHeight, position.z};
    }
};

// -----------------------------------------------------------------------------
// High-End Vulkan Renderer Engine
// -----------------------------------------------------------------------------
class VulkanRenderer {
public:
    explicit VulkanRenderer(ANativeWindow* w) : window_(w) {
        if (window_) ANativeWindow_acquire(window_);
    }

    ~VulkanRenderer() {
        stop();
        if (window_) ANativeWindow_release(window_);
    }

    void start() {
        try {
            initialize();
            running_.store(true);
            started_ = std::chrono::steady_clock::now();
            thread_ = std::thread(&VulkanRenderer::renderLoop, this);
        } catch (const std::exception& e) {
            setStatus(std::string("Vulkan init failed: ") + e.what());
            LOGE("%s", status_.c_str());
            cleanup();
        }
    }

    void resize(int width, int height) {
        // Android can report transient zero-sized surfaces during lifecycle changes.
        if (width > 0 && height > 0) resizeRequested_.store(true);
    }

    void stop() {
        running_.store(false);
        if (thread_.joinable()) thread_.join();
        cleanup();
    }

    std::string status() {
        std::lock_guard<std::mutex> lock(statusMutex_);
        return status_;
    }

    void look(float dx, float dy) {
        yaw_.store(yaw_.load() + dx * 0.0035f);
        pitch_.store(std::clamp(pitch_.load() + dy * 0.0035f, -1.48f, 1.48f));
    }

    void setJoystickInput(float x, float y) {
        inputX_.store(x);
        inputY_.store(y);
    }

    void triggerJump() { jumpRequested_.store(true); }

    void updateSettings(float fog, float timeOfDay, float exp) {
        fogDensity_.store(fog);
        timeOfDay_.store(timeOfDay);
        exposure_.store(exp);
    }

    void resetView() {
        std::lock_guard<std::mutex> lock(physicsMutex_);
        player_.position = {0.0f, 2.0f, 8.0f};
        player_.velocity = {0.0f, 0.0f, 0.0f};
        yaw_.store(0.0f);
        pitch_.store(-0.1f);
    }

    std::string loadGlb(const std::vector<std::uint8_t>& bytes) {
        try {
            auto decoded = assets::readGlbMeshes(bytes);

            std::size_t vertexTotal = 0, indexTotal = 0;
            for (const auto& p : decoded.primitives) {
                vertexTotal += p.vertices.size();
                indexTotal += p.indices.size();
            }
            if (vertexTotal == 0 || indexTotal == 0) throw std::runtime_error("Walang valid mesh primitives ang GLB");

            std::vector<Vertex> vertices; vertices.reserve(vertexTotal);
            std::vector<std::uint32_t> indices; indices.reserve(indexTotal);

            float lo[3] = {1e9f, 1e9f, 1e9f}, hi[3] = {-1e9f, -1e9f, -1e9f};
            for (const auto& p : decoded.primitives) {
                for (const auto& v : p.vertices) {
                    for (int k = 0; k < 3; ++k) {
                        lo[k] = std::min(lo[k], v.position[k]);
                        hi[k] = std::max(hi[k], v.position[k]);
                    }
                }
            }

            const float cx = (lo[0] + hi[0]) * 0.5f;
            const float cy = lo[1];
            const float cz = (lo[2] + hi[2]) * 0.5f;
            const float span = std::max({hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]});
            const float scaleFactor = 4.0f / std::max(span, 1e-4f);

            std::vector<DrawRange> ranges;
            for (const auto& p : decoded.primitives) {
                const auto base = static_cast<std::uint32_t>(vertices.size());
                DrawRange range{};
                range.firstIndex = static_cast<uint32_t>(indices.size());
                range.indexCount = static_cast<uint32_t>(p.indices.size());
                std::copy(p.baseColorFactor, p.baseColorFactor + 4, range.baseColor);
                range.metallic = p.metallicFactor;
                range.roughness = p.roughnessFactor;

                for (const auto& v : p.vertices) {
                    Vertex out{};
                    out.position[0] = (v.position[0] - cx) * scaleFactor;
                    out.position[1] = (v.position[1] - cy) * scaleFactor;
                    out.position[2] = (v.position[2] - cz) * scaleFactor;
                    std::copy(v.normal, v.normal + 3, out.normal);
                    out.uv[0] = 0.0f; out.uv[1] = 0.0f;
                    // Default tangent vector
                    out.tangent[0] = 1.0f; out.tangent[1] = 0.0f; out.tangent[2] = 0.0f; out.tangent[3] = 1.0f;
                    vertices.push_back(out);
                }
                for (auto idx : p.indices) indices.push_back(base + idx);
                ranges.push_back(range);
            }

            std::lock_guard<std::mutex> lock(gpuMutex_);
            if (!device_) throw std::runtime_error("Device not initialized");
            vkDeviceWaitIdle(device_);

            if (glbVertexBuffer_) vkDestroyBuffer(device_, glbVertexBuffer_, nullptr);
            if (glbIndexBuffer_) vkDestroyBuffer(device_, glbIndexBuffer_, nullptr);
            if (glbVertexMemory_) vkFreeMemory(device_, glbVertexMemory_, nullptr);
            if (glbIndexMemory_) vkFreeMemory(device_, glbIndexMemory_, nullptr);

            createBuffer(vertices.size() * sizeof(Vertex), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, glbVertexBuffer_, glbVertexMemory_, vertices.data());
            createBuffer(indices.size() * sizeof(uint32_t), VK_BUFFER_USAGE_INDEX_BUFFER_BIT, glbIndexBuffer_, glbIndexMemory_, indices.data());
            glbRanges_ = std::move(ranges);
            hasGlbModel_ = true;

            setStatus("Ultra GLB Imported (" + std::to_string(vertexTotal) + " verts)");
            LOGI("%s", status_.c_str());
            return status_;
        } catch (const std::exception& e) {
            setStatus(std::string("GLB Error: ") + e.what());
            LOGE("%s", status_.c_str());
            return status_;
        }
    }

private:
    ANativeWindow* window_ = nullptr;
    VkInstance instance_ = VK_NULL_HANDLE;
    VkSurfaceKHR surface_ = VK_NULL_HANDLE;
    VkPhysicalDevice physical_ = VK_NULL_HANDLE;
    VkDevice device_ = VK_NULL_HANDLE;
    uint32_t graphicsFamily_ = UINT32_MAX, presentFamily_ = UINT32_MAX;
    VkQueue graphicsQueue_ = VK_NULL_HANDLE, presentQueue_ = VK_NULL_HANDLE;

    // Swapchain & Primary Render Targets
    VkSwapchainKHR swapchain_ = VK_NULL_HANDLE;
    VkFormat format_ = VK_FORMAT_UNDEFINED, depthFormat_ = VK_FORMAT_D32_SFLOAT;
    VkExtent2D extent_{};
    std::vector<VkImage> images_;
    std::vector<VkImageView> views_;
    std::vector<VkImage> depthImages_;
    std::vector<VkDeviceMemory> depthMemory_;
    std::vector<VkImageView> depthViews_;
    VkRenderPass renderPass_ = VK_NULL_HANDLE;
    std::vector<VkFramebuffer> framebuffers_;

    // Real-Time Directional Shadow Map Pass (Cascaded/Ortho Depth)
    static constexpr uint32_t kShadowMapDim = 2048;
    VkFormat shadowDepthFormat_ = VK_FORMAT_D32_SFLOAT;
    VkImage shadowImage_ = VK_NULL_HANDLE;
    VkDeviceMemory shadowMemory_ = VK_NULL_HANDLE;
    VkImageView shadowView_ = VK_NULL_HANDLE;
    VkSampler shadowSampler_ = VK_NULL_HANDLE;
    VkRenderPass shadowRenderPass_ = VK_NULL_HANDLE;
    VkFramebuffer shadowFramebuffer_ = VK_NULL_HANDLE;
    VkPipelineLayout shadowPipelineLayout_ = VK_NULL_HANDLE;
    VkPipeline shadowPipeline_ = VK_NULL_HANDLE;

    // UBO & Descriptors
    static constexpr size_t kFrames = 2;
    VkDescriptorSetLayout sceneDescLayout_ = VK_NULL_HANDLE;
    VkDescriptorPool descPool_ = VK_NULL_HANDLE;
    std::array<VkBuffer, kFrames> sceneUboBuffers_{};
    std::array<VkDeviceMemory, kFrames> sceneUboMemory_{};
    std::array<void*, kFrames> sceneUboMapped_{};
    std::array<VkDescriptorSet, kFrames> sceneDescSets_{};

    // Forward+ PBR & Sky Pipelines
    VkPipelineLayout pbrPipelineLayout_ = VK_NULL_HANDLE;
    VkPipeline pbrPipeline_ = VK_NULL_HANDLE;
    VkPipelineLayout skyPipelineLayout_ = VK_NULL_HANDLE;
    VkPipeline skyPipeline_ = VK_NULL_HANDLE;

    // High-Resolution Smooth Meshes
    struct RealisticMesh {
        VkBuffer vb = VK_NULL_HANDLE;
        VkDeviceMemory vMem = VK_NULL_HANDLE;
        VkBuffer ib = VK_NULL_HANDLE;
        VkDeviceMemory iMem = VK_NULL_HANDLE;
        uint32_t indexCount = 0;
        Mat4 modelMatrix;
        float baseColor[4]{1, 1, 1, 1};
        float metallic = 0.0f;
        float roughness = 0.5f;
        float isWater = 0.0f;
    };
    std::vector<RealisticMesh> worldMeshes_;

    // Dynamic GLB Mesh
    VkBuffer glbVertexBuffer_ = VK_NULL_HANDLE, glbIndexBuffer_ = VK_NULL_HANDLE;
    VkDeviceMemory glbVertexMemory_ = VK_NULL_HANDLE, glbIndexMemory_ = VK_NULL_HANDLE;
    bool hasGlbModel_ = false;
    struct DrawRange {
        uint32_t firstIndex = 0, indexCount = 0;
        float baseColor[4]{1.0f, 1.0f, 1.0f, 1.0f};
        float metallic = 0.5f, roughness = 0.5f;
    };
    std::vector<DrawRange> glbRanges_;

    // Command & Synchronization
    VkCommandPool commandPool_ = VK_NULL_HANDLE;
    std::vector<VkCommandBuffer> commandBuffers_;
    std::array<VkSemaphore, kFrames> imageAvailable_{};
    std::array<VkSemaphore, kFrames> renderFinished_{};
    std::array<VkFence, kFrames> fences_{};
    size_t frame_ = 0;

    // Controls, Physics, & LifeCycle
    std::atomic<bool> running_{false}, resizeRequested_{false}, jumpRequested_{false};
    std::mutex gpuMutex_, physicsMutex_, lifecycleMutex_, statusMutex_;
    CharacterController player_;
    std::vector<PhysicsObstacle> physicsObstacles_;
    std::atomic<float> yaw_{0.0f}, pitch_{-0.1f};
    std::atomic<float> inputX_{0.0f}, inputY_{0.0f};

    std::atomic<float> fogDensity_{0.025f};
    std::atomic<float> timeOfDay_{14.5f};
    std::atomic<float> exposure_{1.2f};

    std::thread thread_;
    std::chrono::steady_clock::time_point started_{};
    std::string deviceName_ = "GPU", status_ = "Initializing Engine...";

    void setStatus(const std::string& message) {
        std::lock_guard<std::mutex> lock(statusMutex_);
        status_ = message;
    }

    void initialize() {
        VkApplicationInfo app{};
        app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        app.pApplicationName = "Neo-3D AAA";
        app.applicationVersion = VK_MAKE_VERSION(2, 0, 0);
        app.pEngineName = "Neo-3D PBR Forward+";
        app.engineVersion = VK_MAKE_VERSION(2, 0, 0);
        app.apiVersion = VK_API_VERSION_1_0;

        const char* exts[] = {VK_KHR_SURFACE_EXTENSION_NAME, VK_KHR_ANDROID_SURFACE_EXTENSION_NAME};
        VkInstanceCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        ci.pApplicationInfo = &app;
        ci.enabledExtensionCount = 2;
        ci.ppEnabledExtensionNames = exts;
        check(vkCreateInstance(&ci, nullptr, &instance_), "vkCreateInstance");

        VkAndroidSurfaceCreateInfoKHR si{};
        si.sType = VK_STRUCTURE_TYPE_ANDROID_SURFACE_CREATE_INFO_KHR;
        si.window = window_;
        check(vkCreateAndroidSurfaceKHR(instance_, &si, nullptr, &surface_), "vkCreateAndroidSurfaceKHR");

        selectDevice();
        createDevice();
        chooseDepthFormat();
        createSwapchain();
        createRenderPass();
        createDepthResources();
        createFramebuffers();

        // 1. Shadows Pass Setup
        createShadowResources();
        createShadowRenderPass();
        createShadowFramebuffer();

        // 2. Uniform Buffers and Scene Descriptors
        createDescriptorSetLayout();
        createUniformBuffers();
        createDescriptorPool();
        createDescriptorSets();

        // 3. Pipelines
        createCommands();
        createShadowPipeline();
        createPBRPipeline();
        createSkyPipeline();
        createSync();

        // 4. Detailed High-Poly Realistic Scene Geometry
        buildUltraRealisticEnvironment();

        setStatus("Engine Online | " + deviceName_ + " | Forward+ PBR & CSM Shadow Active");
    }

    void selectDevice() {
        uint32_t count = 0;
        check(vkEnumeratePhysicalDevices(instance_, &count, nullptr), "Enumerate physical devices");
        std::vector<VkPhysicalDevice> ds(count);
        check(vkEnumeratePhysicalDevices(instance_, &count, ds.data()), "Get physical devices");

        for (auto d : ds) {
            uint32_t qc = 0;
            vkGetPhysicalDeviceQueueFamilyProperties(d, &qc, nullptr);
            std::vector<VkQueueFamilyProperties> qs(qc);
            vkGetPhysicalDeviceQueueFamilyProperties(d, &qc, qs.data());

            uint32_t gf = UINT32_MAX, pf = UINT32_MAX;
            for (uint32_t i = 0; i < qc; ++i) {
                if (qs[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) gf = i;
                VkBool32 supported = VK_FALSE;
                vkGetPhysicalDeviceSurfaceSupportKHR(d, i, surface_, &supported);
                if (supported) pf = i;
            }
            if (gf != UINT32_MAX && pf != UINT32_MAX) {
                physical_ = d; graphicsFamily_ = gf; presentFamily_ = pf;
                VkPhysicalDeviceProperties p{};
                vkGetPhysicalDeviceProperties(d, &p);
                deviceName_ = p.deviceName;
                return;
            }
        }
        throw std::runtime_error("Walang physical device na sumusuporta sa graphics at present queues");
    }

    void createDevice() {
        float prio = 1.0f;
        std::vector<VkDeviceQueueCreateInfo> qs;
        for (uint32_t f : {graphicsFamily_, presentFamily_}) {
            if (std::any_of(qs.begin(), qs.end(), [f](const auto& q) { return q.queueFamilyIndex == f; })) continue;
            VkDeviceQueueCreateInfo q{};
            q.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
            q.queueFamilyIndex = f;
            q.queueCount = 1;
            q.pQueuePriorities = &prio;
            qs.push_back(q);
        }

        const char* ext = VK_KHR_SWAPCHAIN_EXTENSION_NAME;
        VkPhysicalDeviceFeatures features{};
        features.samplerAnisotropy = VK_FALSE;

        VkDeviceCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        ci.queueCreateInfoCount = static_cast<uint32_t>(qs.size());
        ci.pQueueCreateInfos = qs.data();
        ci.enabledExtensionCount = 1;
        ci.ppEnabledExtensionNames = &ext;
        ci.pEnabledFeatures = &features;
        check(vkCreateDevice(physical_, &ci, nullptr, &device_), "vkCreateDevice");

        vkGetDeviceQueue(device_, graphicsFamily_, 0, &graphicsQueue_);
        vkGetDeviceQueue(device_, presentFamily_, 0, &presentQueue_);
    }

    uint32_t memoryType(uint32_t bits, VkMemoryPropertyFlags flags) {
        VkPhysicalDeviceMemoryProperties p{};
        vkGetPhysicalDeviceMemoryProperties(physical_, &p);
        for (uint32_t i = 0; i < p.memoryTypeCount; ++i) {
            if ((bits & (1u << i)) && (p.memoryTypes[i].propertyFlags & flags) == flags) return i;
        }
        throw std::runtime_error("Walang angkop na memory type para sa allocation");
    }

    void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkBuffer& buffer, VkDeviceMemory& mem, const void* data) {
        VkBufferCreateInfo bi{};
        bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bi.size = size;
        bi.usage = usage;
        bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        check(vkCreateBuffer(device_, &bi, nullptr, &buffer), "vkCreateBuffer");

        VkMemoryRequirements req{};
        vkGetBufferMemoryRequirements(device_, buffer, &req);

        VkMemoryAllocateInfo ai{};
        ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        ai.allocationSize = req.size;
        ai.memoryTypeIndex = memoryType(req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        check(vkAllocateMemory(device_, &ai, nullptr, &mem), "vkAllocateMemory");
        check(vkBindBufferMemory(device_, buffer, mem, 0), "vkBindBufferMemory");

        if (data) {
            void* mapped = nullptr;
            check(vkMapMemory(device_, mem, 0, size, 0, &mapped), "vkMapMemory");
            std::memcpy(mapped, data, static_cast<size_t>(size));
            vkUnmapMemory(device_, mem);
        }
    }

    void chooseDepthFormat() {
        for (VkFormat f : {VK_FORMAT_D32_SFLOAT, VK_FORMAT_D24_UNORM_S8_UINT, VK_FORMAT_D16_UNORM}) {
            VkFormatProperties p{};
            vkGetPhysicalDeviceFormatProperties(physical_, f, &p);
            if (p.optimalTilingFeatures & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT) {
                depthFormat_ = f;
                shadowDepthFormat_ = f;
                return;
            }
        }
        throw std::runtime_error("Walang depth format na available");
    }

    void createSwapchain() {
        VkSurfaceCapabilitiesKHR caps{};
        check(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physical_, surface_, &caps), "vkGetCaps");

        uint32_t n = 0;
        vkGetPhysicalDeviceSurfaceFormatsKHR(physical_, surface_, &n, nullptr);
        std::vector<VkSurfaceFormatKHR> fs(n);
        vkGetPhysicalDeviceSurfaceFormatsKHR(physical_, surface_, &n, fs.data());

        format_ = fs[0].format;
        for (const auto& f : fs) {
            if (f.format == VK_FORMAT_R8G8B8A8_UNORM || f.format == VK_FORMAT_B8G8R8A8_UNORM) {
                format_ = f.format; break;
            }
        }

        if (caps.currentExtent.width != UINT32_MAX) {
            extent_ = caps.currentExtent;
        } else {
            extent_.width = std::clamp(static_cast<uint32_t>(ANativeWindow_getWidth(window_)), caps.minImageExtent.width, caps.maxImageExtent.width);
            extent_.height = std::clamp(static_cast<uint32_t>(ANativeWindow_getHeight(window_)), caps.minImageExtent.height, caps.maxImageExtent.height);
        }

        uint32_t ic = caps.minImageCount + 1;
        if (caps.maxImageCount && ic > caps.maxImageCount) ic = caps.maxImageCount;

        VkSwapchainCreateInfoKHR ci{};
        ci.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
        ci.surface = surface_;
        ci.minImageCount = ic;
        ci.imageFormat = format_;
        ci.imageColorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
        ci.imageExtent = extent_;
        ci.imageArrayLayers = 1;
        ci.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
        ci.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
        ci.preTransform = caps.currentTransform;
        ci.compositeAlpha = VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR;
        ci.presentMode = VK_PRESENT_MODE_FIFO_KHR;
        ci.clipped = VK_TRUE;

        check(vkCreateSwapchainKHR(device_, &ci, nullptr, &swapchain_), "vkCreateSwapchainKHR");

        uint32_t actual = 0;
        vkGetSwapchainImagesKHR(device_, swapchain_, &actual, nullptr);
        images_.resize(actual);
        vkGetSwapchainImagesKHR(device_, swapchain_, &actual, images_.data());

        views_.resize(images_.size());
        for (size_t i = 0; i < images_.size(); ++i) {
            VkImageViewCreateInfo vi{};
            vi.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            vi.image = images_[i];
            vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
            vi.format = format_;
            vi.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            check(vkCreateImageView(device_, &vi, nullptr, &views_[i]), "vkCreateImageView");
        }
    }

    void createRenderPass() {
        std::array<VkAttachmentDescription, 2> a{};
        a[0].format = format_;
        a[0].samples = VK_SAMPLE_COUNT_1_BIT;
        a[0].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        a[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        a[0].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        a[0].finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

        a[1].format = depthFormat_;
        a[1].samples = VK_SAMPLE_COUNT_1_BIT;
        a[1].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        a[1].storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        a[1].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        a[1].finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

        VkAttachmentReference cRef{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
        VkAttachmentReference dRef{1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};

        VkSubpassDescription sub{};
        sub.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        sub.colorAttachmentCount = 1;
        sub.pColorAttachments = &cRef;
        sub.pDepthStencilAttachment = &dRef;

        VkSubpassDependency dep{};
        dep.srcSubpass = VK_SUBPASS_EXTERNAL;
        dep.dstSubpass = 0;
        dep.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
        dep.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
        dep.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

        VkRenderPassCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        ci.attachmentCount = 2;
        ci.pAttachments = a.data();
        ci.subpassCount = 1;
        ci.pSubpasses = &sub;
        ci.dependencyCount = 1;
        ci.pDependencies = &dep;
        check(vkCreateRenderPass(device_, &ci, nullptr, &renderPass_), "vkCreateRenderPass");
    }

    void createDepthResources() {
        depthImages_.resize(images_.size());
        depthMemory_.resize(images_.size());
        depthViews_.resize(images_.size());

        for (size_t i = 0; i < images_.size(); ++i) {
            VkImageCreateInfo ii{};
            ii.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
            ii.imageType = VK_IMAGE_TYPE_2D;
            ii.extent = {extent_.width, extent_.height, 1};
            ii.mipLevels = 1; ii.arrayLayers = 1;
            ii.format = depthFormat_;
            ii.tiling = VK_IMAGE_TILING_OPTIMAL;
            ii.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
            ii.samples = VK_SAMPLE_COUNT_1_BIT;
            check(vkCreateImage(device_, &ii, nullptr, &depthImages_[i]), "vkCreateImage depth");

            VkMemoryRequirements req{};
            vkGetImageMemoryRequirements(device_, depthImages_[i], &req);

            VkMemoryAllocateInfo ai{};
            ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
            ai.allocationSize = req.size;
            ai.memoryTypeIndex = memoryType(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
            check(vkAllocateMemory(device_, &ai, nullptr, &depthMemory_[i]), "vkAllocateMemory depth");
            check(vkBindImageMemory(device_, depthImages_[i], depthMemory_[i], 0), "vkBindImageMemory depth");

            VkImageViewCreateInfo vi{};
            vi.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            vi.image = depthImages_[i];
            vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
            vi.format = depthFormat_;
            vi.subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1};
            check(vkCreateImageView(device_, &vi, nullptr, &depthViews_[i]), "vkCreateImageView depth");
        }
    }

    void createFramebuffers() {
        framebuffers_.resize(views_.size());
        for (size_t i = 0; i < views_.size(); ++i) {
            VkImageView atts[] = {views_[i], depthViews_[i]};
            VkFramebufferCreateInfo ci{};
            ci.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
            ci.renderPass = renderPass_;
            ci.attachmentCount = 2;
            ci.pAttachments = atts;
            ci.width = extent_.width;
            ci.height = extent_.height;
            ci.layers = 1;
            check(vkCreateFramebuffer(device_, &ci, nullptr, &framebuffers_[i]), "vkCreateFramebuffer");
        }
    }

    // -------------------------------------------------------------------------
    // Dedicated Shadow Pass Infrastructure
    // -------------------------------------------------------------------------
    void createShadowResources() {
        VkImageCreateInfo ii{};
        ii.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        ii.imageType = VK_IMAGE_TYPE_2D;
        ii.extent = {kShadowMapDim, kShadowMapDim, 1};
        ii.mipLevels = 1;
        ii.arrayLayers = 1;
        ii.format = shadowDepthFormat_;
        ii.tiling = VK_IMAGE_TILING_OPTIMAL;
        ii.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        ii.samples = VK_SAMPLE_COUNT_1_BIT;
        check(vkCreateImage(device_, &ii, nullptr, &shadowImage_), "Create shadow image");

        VkMemoryRequirements req{};
        vkGetImageMemoryRequirements(device_, shadowImage_, &req);

        VkMemoryAllocateInfo ai{};
        ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        ai.allocationSize = req.size;
        ai.memoryTypeIndex = memoryType(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        check(vkAllocateMemory(device_, &ai, nullptr, &shadowMemory_), "Alloc shadow mem");
        check(vkBindImageMemory(device_, shadowImage_, shadowMemory_, 0), "Bind shadow mem");

        VkImageViewCreateInfo vi{};
        vi.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        vi.image = shadowImage_;
        vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
        vi.format = shadowDepthFormat_;
        vi.subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1};
        check(vkCreateImageView(device_, &vi, nullptr, &shadowView_), "Create shadow view");

        // High Quality Hardware PCF Sampler
        VkSamplerCreateInfo sci{};
        sci.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        sci.magFilter = VK_FILTER_LINEAR;
        sci.minFilter = VK_FILTER_LINEAR;
        sci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
        sci.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
        sci.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
        sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
        sci.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
        sci.compareEnable = VK_TRUE;
        sci.compareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
        check(vkCreateSampler(device_, &sci, nullptr, &shadowSampler_), "Create shadow sampler");
    }

    void createShadowRenderPass() {
        VkAttachmentDescription att{};
        att.format = shadowDepthFormat_;
        att.samples = VK_SAMPLE_COUNT_1_BIT;
        att.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        att.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        att.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        att.finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        VkAttachmentReference ref{0, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};
        VkSubpassDescription sub{};
        sub.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        sub.pDepthStencilAttachment = &ref;

        VkSubpassDependency dep{};
        dep.srcSubpass = 0;
        dep.dstSubpass = VK_SUBPASS_EXTERNAL;
        dep.srcStageMask = VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
        dep.dstStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
        dep.srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        dep.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

        VkRenderPassCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        ci.attachmentCount = 1;
        ci.pAttachments = &att;
        ci.subpassCount = 1;
        ci.pSubpasses = &sub;
        ci.dependencyCount = 1;
        ci.pDependencies = &dep;
        check(vkCreateRenderPass(device_, &ci, nullptr, &shadowRenderPass_), "Create shadow render pass");
    }

    void createShadowFramebuffer() {
        VkFramebufferCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        ci.renderPass = shadowRenderPass_;
        ci.attachmentCount = 1;
        ci.pAttachments = &shadowView_;
        ci.width = kShadowMapDim;
        ci.height = kShadowMapDim;
        ci.layers = 1;
        check(vkCreateFramebuffer(device_, &ci, nullptr, &shadowFramebuffer_), "Create shadow fb");
    }

    // -------------------------------------------------------------------------
    // Descriptors & Uniform Buffers
    // -------------------------------------------------------------------------
    void createDescriptorSetLayout() {
        VkDescriptorSetLayoutBinding bUbo{};
        bUbo.binding = 0;
        bUbo.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        bUbo.descriptorCount = 1;
        bUbo.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;

        VkDescriptorSetLayoutBinding bShadow{};
        bShadow.binding = 1;
        bShadow.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        bShadow.descriptorCount = 1;
        bShadow.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

        std::array<VkDescriptorSetLayoutBinding, 2> bindings = {bUbo, bShadow};
        VkDescriptorSetLayoutCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        ci.bindingCount = static_cast<uint32_t>(bindings.size());
        ci.pBindings = bindings.data();
        check(vkCreateDescriptorSetLayout(device_, &ci, nullptr, &sceneDescLayout_), "Create desc layout");
    }

    void createUniformBuffers() {
        VkDeviceSize sz = sizeof(SceneUniformBuffer);
        for (size_t i = 0; i < kFrames; ++i) {
            createBuffer(sz, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, sceneUboBuffers_[i], sceneUboMemory_[i], nullptr);
            check(vkMapMemory(device_, sceneUboMemory_[i], 0, sz, 0, &sceneUboMapped_[i]), "Map UBO");
        }
    }

    void createDescriptorPool() {
        std::array<VkDescriptorPoolSize, 2> ps{};
        ps[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        ps[0].descriptorCount = static_cast<uint32_t>(kFrames);
        ps[1].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        ps[1].descriptorCount = static_cast<uint32_t>(kFrames);

        VkDescriptorPoolCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        ci.maxSets = static_cast<uint32_t>(kFrames);
        ci.poolSizeCount = static_cast<uint32_t>(ps.size());
        ci.pPoolSizes = ps.data();
        check(vkCreateDescriptorPool(device_, &ci, nullptr, &descPool_), "Create desc pool");
    }

    void createDescriptorSets() {
        std::vector<VkDescriptorSetLayout> layouts(kFrames, sceneDescLayout_);
        VkDescriptorSetAllocateInfo ai{};
        ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        ai.descriptorPool = descPool_;
        ai.descriptorSetCount = static_cast<uint32_t>(kFrames);
        ai.pSetLayouts = layouts.data();
        check(vkAllocateDescriptorSets(device_, &ai, sceneDescSets_.data()), "Alloc desc sets");

        for (size_t i = 0; i < kFrames; ++i) {
            VkDescriptorBufferInfo bi{};
            bi.buffer = sceneUboBuffers_[i];
            bi.offset = 0;
            bi.range = sizeof(SceneUniformBuffer);

            VkDescriptorImageInfo ii{};
            ii.sampler = shadowSampler_;
            ii.imageView = shadowView_;
            ii.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

            std::array<VkWriteDescriptorSet, 2> ws{};
            ws[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            ws[0].dstSet = sceneDescSets_[i];
            ws[0].dstBinding = 0;
            ws[0].descriptorCount = 1;
            ws[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            ws[0].pBufferInfo = &bi;

            ws[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            ws[1].dstSet = sceneDescSets_[i];
            ws[1].dstBinding = 1;
            ws[1].descriptorCount = 1;
            ws[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            ws[1].pImageInfo = &ii;

            vkUpdateDescriptorSets(device_, static_cast<uint32_t>(ws.size()), ws.data(), 0, nullptr);
        }
    }

    VkShaderModule createShader(const uint32_t* code, size_t sz) {
        VkShaderModuleCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        ci.codeSize = sz;
        ci.pCode = code;
        VkShaderModule m = VK_NULL_HANDLE;
        check(vkCreateShaderModule(device_, &ci, nullptr, &m), "vkCreateShaderModule");
        return m;
    }

    // -------------------------------------------------------------------------
    // Pipelines
    // -------------------------------------------------------------------------
    void createShadowPipeline() {
        // Shadow pass push constant (Light MVP + Model)
        VkPushConstantRange range{};
        range.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
        range.offset = 0;
        range.size = sizeof(Mat4);

        VkPipelineLayoutCreateInfo li{};
        li.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        li.pushConstantRangeCount = 1;
        li.pPushConstantRanges = &range;
        check(vkCreatePipelineLayout(device_, &li, nullptr, &shadowPipelineLayout_), "Shadow layout");

        VkShaderModule v = createShader(kShadowVert, kShadowVertSize);

        VkPipelineShaderStageCreateInfo stage{};
        stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stage.stage = VK_SHADER_STAGE_VERTEX_BIT;
        stage.module = v;
        stage.pName = "main";

        VkVertexInputBindingDescription b{};
        b.binding = 0;
        b.stride = sizeof(Vertex);
        b.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

        VkVertexInputAttributeDescription a{};
        a.location = 0;
        a.binding = 0;
        a.format = VK_FORMAT_R32G32B32_SFLOAT;
        a.offset = offsetof(Vertex, position);

        VkPipelineVertexInputStateCreateInfo vi{};
        vi.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        vi.vertexBindingDescriptionCount = 1;
        vi.pVertexBindingDescriptions = &b;
        vi.vertexAttributeDescriptionCount = 1;
        vi.pVertexAttributeDescriptions = &a;

        VkPipelineInputAssemblyStateCreateInfo ia{};
        ia.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

        VkViewport vp{0.0f, 0.0f, static_cast<float>(kShadowMapDim), static_cast<float>(kShadowMapDim), 0.0f, 1.0f};
        VkRect2D sc{{0, 0}, {kShadowMapDim, kShadowMapDim}};
        VkPipelineViewportStateCreateInfo vsi{};
        vsi.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        vsi.viewportCount = 1; vsi.pViewports = &vp;
        vsi.scissorCount = 1;  vsi.pScissors = &sc;

        VkPipelineRasterizationStateCreateInfo rs{};
        rs.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        rs.polygonMode = VK_POLYGON_MODE_FILL;
        rs.cullMode = VK_CULL_MODE_NONE;
        rs.depthBiasEnable = VK_TRUE;
        rs.depthBiasConstantFactor = 1.25f;
        rs.depthBiasSlopeFactor = 1.75f;
        rs.lineWidth = 1.0f;

        VkPipelineMultisampleStateCreateInfo ms{};
        ms.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

        VkPipelineDepthStencilStateCreateInfo ds{};
        ds.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
        ds.depthTestEnable = VK_TRUE;
        ds.depthWriteEnable = VK_TRUE;
        ds.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;

        VkGraphicsPipelineCreateInfo pi{};
        pi.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        pi.stageCount = 1;
        pi.pStages = &stage;
        pi.pVertexInputState = &vi;
        pi.pInputAssemblyState = &ia;
        pi.pViewportState = &vsi;
        pi.pRasterizationState = &rs;
        pi.pMultisampleState = &ms;
        pi.pDepthStencilState = &ds;
        pi.layout = shadowPipelineLayout_;
        pi.renderPass = shadowRenderPass_;

        check(vkCreateGraphicsPipelines(device_, VK_NULL_HANDLE, 1, &pi, nullptr, &shadowPipeline_), "Create shadow pipe");
        vkDestroyShaderModule(device_, v, nullptr);
    }

    void createPBRPipeline() {
        VkPushConstantRange range{};
        range.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
        range.offset = 0;
        range.size = sizeof(ObjectPushConstants);

        VkPipelineLayoutCreateInfo li{};
        li.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        li.setLayoutCount = 1;
        li.pSetLayouts = &sceneDescLayout_;
        li.pushConstantRangeCount = 1;
        li.pPushConstantRanges = &range;
        check(vkCreatePipelineLayout(device_, &li, nullptr, &pbrPipelineLayout_), "Create PBR layout");

        VkShaderModule v = createShader(kMeshVert, kMeshVertSize);
        VkShaderModule f = createShader(kMeshFrag, kMeshFragSize);

        VkPipelineShaderStageCreateInfo stages[2]{};
        stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
        stages[0].module = v; stages[0].pName = "main";
        stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        stages[1].module = f; stages[1].pName = "main";

        VkVertexInputBindingDescription b{};
        b.binding = 0;
        b.stride = sizeof(Vertex);
        b.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

        std::array<VkVertexInputAttributeDescription, 4> a{};
        // Position
        a[0].location = 0; a[0].binding = 0; a[0].format = VK_FORMAT_R32G32B32_SFLOAT; a[0].offset = offsetof(Vertex, position);
        // Normal
        a[1].location = 1; a[1].binding = 0; a[1].format = VK_FORMAT_R32G32B32_SFLOAT; a[1].offset = offsetof(Vertex, normal);
        // UV
        a[2].location = 2; a[2].binding = 0; a[2].format = VK_FORMAT_R32G32_SFLOAT;    a[2].offset = offsetof(Vertex, uv);
        // Tangent
        a[3].location = 3; a[3].binding = 0; a[3].format = VK_FORMAT_R32G32B32A32_SFLOAT; a[3].offset = offsetof(Vertex, tangent);

        VkPipelineVertexInputStateCreateInfo vi{};
        vi.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        vi.vertexBindingDescriptionCount = 1;
        vi.pVertexBindingDescriptions = &b;
        vi.vertexAttributeDescriptionCount = static_cast<uint32_t>(a.size());
        vi.pVertexAttributeDescriptions = a.data();

        VkPipelineInputAssemblyStateCreateInfo ia{};
        ia.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

        VkViewport vp{0.0f, 0.0f, static_cast<float>(extent_.width), static_cast<float>(extent_.height), 0.0f, 1.0f};
        VkRect2D sc{{0, 0}, extent_};
        VkPipelineViewportStateCreateInfo vsi{};
        vsi.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        vsi.viewportCount = 1; vsi.pViewports = &vp;
        vsi.scissorCount = 1;  vsi.pScissors = &sc;

        VkPipelineRasterizationStateCreateInfo rs{};
        rs.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        rs.polygonMode = VK_POLYGON_MODE_FILL;
        rs.cullMode = VK_CULL_MODE_BACK_BIT;
        rs.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        rs.lineWidth = 1.0f;

        VkPipelineMultisampleStateCreateInfo ms{};
        ms.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

        VkPipelineDepthStencilStateCreateInfo ds{};
        ds.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
        ds.depthTestEnable = VK_TRUE;
        ds.depthWriteEnable = VK_TRUE;
        ds.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;

        VkPipelineColorBlendAttachmentState ba{};
        ba.colorWriteMask = 0xf;
        ba.blendEnable = VK_FALSE;

        VkPipelineColorBlendStateCreateInfo bs{};
        bs.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        bs.attachmentCount = 1; bs.pAttachments = &ba;

        VkGraphicsPipelineCreateInfo pi{};
        pi.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        pi.stageCount = 2; pi.pStages = stages;
        pi.pVertexInputState = &vi;
        pi.pInputAssemblyState = &ia;
        pi.pViewportState = &vsi;
        pi.pRasterizationState = &rs;
        pi.pMultisampleState = &ms;
        pi.pDepthStencilState = &ds;
        pi.pColorBlendState = &bs;
        pi.layout = pbrPipelineLayout_;
        pi.renderPass = renderPass_;

        check(vkCreateGraphicsPipelines(device_, VK_NULL_HANDLE, 1, &pi, nullptr, &pbrPipeline_), "vkCreateGraphicsPipelines PBR");
        vkDestroyShaderModule(device_, v, nullptr);
        vkDestroyShaderModule(device_, f, nullptr);
    }

    void createSkyPipeline() {
        VkPushConstantRange range{};
        range.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
        range.offset = 0;
        range.size = sizeof(SkyPushConstants);

        VkPipelineLayoutCreateInfo li{};
        li.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        li.pushConstantRangeCount = 1;
        li.pPushConstantRanges = &range;
        check(vkCreatePipelineLayout(device_, &li, nullptr, &skyPipelineLayout_), "Sky layout");

        VkShaderModule v = createShader(kSkyVert, kSkyVertSize);
        VkShaderModule f = createShader(kSkyFrag, kSkyFragSize);

        VkPipelineShaderStageCreateInfo stages[2]{};
        stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
        stages[0].module = v; stages[0].pName = "main";
        stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        stages[1].module = f; stages[1].pName = "main";

        VkPipelineVertexInputStateCreateInfo vi{};
        vi.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        VkPipelineInputAssemblyStateCreateInfo ia{};
        ia.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

        VkViewport vp{0.0f, 0.0f, static_cast<float>(extent_.width), static_cast<float>(extent_.height), 0.0f, 1.0f};
        VkRect2D sc{{0, 0}, extent_};
        VkPipelineViewportStateCreateInfo vsi{};
        vsi.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        vsi.viewportCount = 1; vsi.pViewports = &vp;
        vsi.scissorCount = 1;  vsi.pScissors = &sc;

        VkPipelineRasterizationStateCreateInfo rs{};
        rs.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        rs.polygonMode = VK_POLYGON_MODE_FILL;
        rs.cullMode = VK_CULL_MODE_NONE;

        VkPipelineMultisampleStateCreateInfo ms{};
        ms.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

        VkPipelineDepthStencilStateCreateInfo ds{};
        ds.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
        ds.depthTestEnable = VK_FALSE;
        ds.depthWriteEnable = VK_FALSE;

        VkPipelineColorBlendAttachmentState ba{};
        ba.colorWriteMask = 0xf;
        VkPipelineColorBlendStateCreateInfo bs{};
        bs.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        bs.attachmentCount = 1; bs.pAttachments = &ba;

        VkGraphicsPipelineCreateInfo pi{};
        pi.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        pi.stageCount = 2; pi.pStages = stages;
        pi.pVertexInputState = &vi;
        pi.pInputAssemblyState = &ia;
        pi.pViewportState = &vsi;
        pi.pRasterizationState = &rs;
        pi.pMultisampleState = &ms;
        pi.pDepthStencilState = &ds;
        pi.pColorBlendState = &bs;
        pi.layout = skyPipelineLayout_;
        pi.renderPass = renderPass_;

        check(vkCreateGraphicsPipelines(device_, VK_NULL_HANDLE, 1, &pi, nullptr, &skyPipeline_), "Create sky pipe");
        vkDestroyShaderModule(device_, v, nullptr);
        vkDestroyShaderModule(device_, f, nullptr);
    }

    // -------------------------------------------------------------------------
    // Ultra Realistic Mesh Generators (Subdivided Terrain, Water, Pillars)
    // -------------------------------------------------------------------------
    RealisticMesh createSubdividedPlane(float size, int divisions, float heightScale, float metallic, float roughness, const float color[4]) {
        std::vector<Vertex> verts;
        std::vector<uint32_t> inds;
        verts.reserve((divisions + 1) * (divisions + 1));
        inds.reserve(divisions * divisions * 6);

        float step = size / static_cast<float>(divisions);
        float half = size * 0.5f;

        for (int z = 0; z <= divisions; ++z) {
            for (int x = 0; x <= divisions; ++x) {
                float px = -half + x * step;
                float pz = -half + z * step;

                // Smooth organic undulating terrain formula
                float py = 0.0f;
                if (heightScale > 0.001f) {
                    py = (std::sin(px * 0.12f) * std::cos(pz * 0.12f) +
                          std::sin(px * 0.25f + 1.2f) * 0.4f) * heightScale;
                }

                // Analytic Normal calculation
                float dHdX = (0.12f * std::cos(px * 0.12f) * std::cos(pz * 0.12f) +
                              0.10f * std::cos(px * 0.25f + 1.2f)) * heightScale;
                float dHdZ = (-0.12f * std::sin(px * 0.12f) * std::sin(pz * 0.12f)) * heightScale;

                Vec3 norm = Vec3{-dHdX, 1.0f, -dHdZ}.normalized();

                Vertex v{};
                v.position[0] = px; v.position[1] = py; v.position[2] = pz;
                v.normal[0] = norm.x; v.normal[1] = norm.y; v.normal[2] = norm.z;
                v.uv[0] = (px + half) * 0.2f;
                v.uv[1] = (pz + half) * 0.2f;

                // Generated Tangent
                Vec3 tang = Vec3{1.0f, dHdX, 0.0f}.normalized();
                v.tangent[0] = tang.x; v.tangent[1] = tang.y; v.tangent[2] = tang.z; v.tangent[3] = 1.0f;

                verts.push_back(v);
            }
        }

        for (int z = 0; z < divisions; ++z) {
            for (int x = 0; x < divisions; ++x) {
                uint32_t i0 = z * (divisions + 1) + x;
                uint32_t i1 = i0 + 1;
                uint32_t i2 = (z + 1) * (divisions + 1) + x;
                uint32_t i3 = i2 + 1;

                inds.push_back(i0); inds.push_back(i2); inds.push_back(i1);
                inds.push_back(i1); inds.push_back(i2); inds.push_back(i3);
            }
        }

        RealisticMesh m;
        m.indexCount = static_cast<uint32_t>(inds.size());
        m.modelMatrix = identity();
        std::copy(color, color + 4, m.baseColor);
        m.metallic = metallic;
        m.roughness = roughness;

        createBuffer(verts.size() * sizeof(Vertex), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, m.vb, m.vMem, verts.data());
        createBuffer(inds.size() * sizeof(uint32_t), VK_BUFFER_USAGE_INDEX_BUFFER_BIT, m.ib, m.iMem, inds.data());
        return m;
    }

    RealisticMesh createSmoothCylinder(float radius, float height, int segments, float metallic, float roughness, const float color[4]) {
        std::vector<Vertex> verts;
        std::vector<uint32_t> inds;

        for (int i = 0; i <= segments; ++i) {
            float theta = (static_cast<float>(i) / segments) * 6.2831853f;
            float cosT = std::cos(theta);
            float sinT = std::sin(theta);

            Vertex vTop{};
            vTop.position[0] = radius * cosT; vTop.position[1] = height; vTop.position[2] = radius * sinT;
            vTop.normal[0] = cosT; vTop.normal[1] = 0.0f; vTop.normal[2] = sinT;
            vTop.uv[0] = static_cast<float>(i) / segments; vTop.uv[1] = 1.0f;
            vTop.tangent[0] = -sinT; vTop.tangent[1] = 0.0f; vTop.tangent[2] = cosT; vTop.tangent[3] = 1.0f;
            verts.push_back(vTop);

            Vertex vBot{};
            vBot.position[0] = radius * cosT; vBot.position[1] = 0.0f; vBot.position[2] = radius * sinT;
            vBot.normal[0] = cosT; vBot.normal[1] = 0.0f; vBot.normal[2] = sinT;
            vBot.uv[0] = static_cast<float>(i) / segments; vBot.uv[1] = 0.0f;
            vBot.tangent[0] = -sinT; vBot.tangent[1] = 0.0f; vBot.tangent[2] = cosT; vBot.tangent[3] = 1.0f;
            verts.push_back(vBot);
        }

        for (int i = 0; i < segments; ++i) {
            uint32_t topA = i * 2;
            uint32_t botA = topA + 1;
            uint32_t topB = (i + 1) * 2;
            uint32_t botB = topB + 1;

            inds.push_back(topA); inds.push_back(botA); inds.push_back(topB);
            inds.push_back(topB); inds.push_back(botA); inds.push_back(botB);
        }

        RealisticMesh m;
        m.indexCount = static_cast<uint32_t>(inds.size());
        m.modelMatrix = identity();
        std::copy(color, color + 4, m.baseColor);
        m.metallic = metallic;
        m.roughness = roughness;

        createBuffer(verts.size() * sizeof(Vertex), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, m.vb, m.vMem, verts.data());
        createBuffer(inds.size() * sizeof(uint32_t), VK_BUFFER_USAGE_INDEX_BUFFER_BIT, m.ib, m.iMem, inds.data());
        return m;
    }

    void buildUltraRealisticEnvironment() {
        worldMeshes_.clear();
        physicsObstacles_.clear();

        // 1. High-Detail Subdivided Terrain (Organic curves, hindi low-poly)
        float terrainCol[4] = {0.35f, 0.42f, 0.28f, 1.0f};
        worldMeshes_.push_back(createSubdividedPlane(160.0f, 96, 1.8f, 0.04f, 0.88f, terrainCol));

        // 2. High-Subdivision Water Lake Surface
        float waterCol[4] = {0.08f, 0.45f, 0.72f, 0.95f};
        RealisticMesh water = createSubdividedPlane(70.0f, 64, 0.0f, 0.05f, 0.02f, waterCol);
        water.modelMatrix = translate(15.0f, 0.2f, -10.0f);
        water.isWater = 1.0f;
        worldMeshes_.push_back(water);

        // 3. Ultra Smooth Architectural Pillars (32 Segments curved pillars)
        float chrome[4] = {0.95f, 0.95f, 0.96f, 1.0f};
        RealisticMesh p1 = createSmoothCylinder(0.85f, 12.0f, 32, 0.98f, 0.08f, chrome);
        p1.modelMatrix = translate(-8.0f, 0.0f, -6.0f);
        worldMeshes_.push_back(p1);

        float gold[4] = {1.0f, 0.82f, 0.25f, 1.0f};
        RealisticMesh p2 = createSmoothCylinder(0.85f, 12.0f, 32, 0.95f, 0.12f, gold);
        p2.modelMatrix = translate(8.0f, 0.0f, -6.0f);
        worldMeshes_.push_back(p2);

        // 4. Register Physics Obstacles para sa character movement
        physicsObstacles_.push_back({{{-9.0f, 0.0f, -7.0f}, {-7.0f, 12.0f, -5.0f}}, false});
        physicsObstacles_.push_back({{{7.0f, 0.0f, -7.0f}, {9.0f, 12.0f, -5.0f}}, false});
    }

    void createCommands() {
        VkCommandPoolCreateInfo pi{};
        pi.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        pi.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        pi.queueFamilyIndex = graphicsFamily_;
        check(vkCreateCommandPool(device_, &pi, nullptr, &commandPool_), "Command pool");

        commandBuffers_.resize(framebuffers_.size());
        VkCommandBufferAllocateInfo ai{};
        ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        ai.commandPool = commandPool_;
        ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        ai.commandBufferCount = static_cast<uint32_t>(commandBuffers_.size());
        check(vkAllocateCommandBuffers(device_, &ai, commandBuffers_.data()), "Alloc commands");
    }

    void createSync() {
        VkSemaphoreCreateInfo si{};
        si.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
        VkFenceCreateInfo fi{};
        fi.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fi.flags = VK_FENCE_CREATE_SIGNALED_BIT;
        for (size_t i = 0; i < kFrames; ++i) {
            check(vkCreateSemaphore(device_, &si, nullptr, &imageAvailable_[i]), "semA");
            check(vkCreateSemaphore(device_, &si, nullptr, &renderFinished_[i]), "semR");
            check(vkCreateFence(device_, &fi, nullptr, &fences_[i]), "fence");
        }
    }

    // -------------------------------------------------------------------------
    // Frame Execution & Render Loop
    // -------------------------------------------------------------------------
    void record(uint32_t imageIndex, float dt) {
        VkCommandBuffer cmd = commandBuffers_[imageIndex];
        VkCommandBufferBeginInfo bi{};
        bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        check(vkBeginCommandBuffer(cmd, &bi), "Begin command buffer");

        float t = std::chrono::duration<float>(std::chrono::steady_clock::now() - started_).count();

        // 1. Character & Physics Update
        {
            std::lock_guard<std::mutex> lock(physicsMutex_);
            if (jumpRequested_.exchange(false)) player_.jump();
            player_.update(dt, inputX_.load(), inputY_.load(), yaw_.load(), physicsObstacles_);
        }

        // 2. Astronomical Sun Position at Lighting Calculation
        float timeOfDay = timeOfDay_.load();
        float sunAngle = (timeOfDay / 24.0f) * 6.2831853f - 1.5707963f;
        float sunElev = std::sin(sunAngle);
        Vec3 sunDir = Vec3{std::cos(sunAngle), std::max(sunElev, 0.05f), 0.35f}.normalized();

        // 3. Directional Sun Matrix (Light Space View-Projection)
        Vec3 eye = player_.getEyePosition();
        Vec3 lightTarget = eye;
        Vec3 lightPos = lightTarget + sunDir * 55.0f;
        Mat4 lightView = lookAt(lightPos, lightTarget, {0.0f, 1.0f, 0.0f});
        Mat4 lightProj = ortho(-40.0f, 40.0f, -40.0f, 40.0f, 1.0f, 150.0f);
        Mat4 shadowViewProj = multiply(lightProj, lightView);

        // 4. Camera View-Projection
        float yaw = yaw_.load();
        float pitch = pitch_.load();
        Vec3 forward = {
            std::sin(yaw) * std::cos(pitch),
            -std::sin(pitch),
            -std::cos(yaw) * std::cos(pitch)
        };
        Mat4 view = lookAt(eye, eye + forward, {0.0f, 1.0f, 0.0f});
        Mat4 proj = perspective(static_cast<float>(extent_.width) / static_cast<float>(std::max(1u, extent_.height)));
        Mat4 viewProj = multiply(proj, view);

        // 5. Update Uniform Buffer (UE4/Unity Scene Params)
        SceneUniformBuffer uboData{};
        uboData.viewProj = viewProj;
        uboData.invViewProj = inverseMat4(viewProj);
        uboData.shadowViewProj = shadowViewProj;
        uboData.cameraPosition = {eye.x, eye.y, eye.z, t};
        uboData.sunDirection = {sunDir.x, sunDir.y, sunDir.z, std::max(sunElev * 3.5f, 0.0f)};

        if (sunElev > 0.0f) {
            uboData.sunColor = {1.0f, 0.94f, 0.86f, 0.25f * sunElev}; // Direct Sun Light
        } else {
            uboData.sunColor = {0.1f, 0.15f, 0.25f, 0.05f};           // Night Ambient Moon
        }
        uboData.fogAtmosphereParams = {fogDensity_.load(), timeOfDay, 0.005f, 0.003f};
        uboData.cloudParameters = {0.62f, 0.8f, 120.0f, 0.45f};
        uboData.renderingSettings = {exposure_.load(), 0.0005f, 1.5f, 0.0f};

        std::memcpy(sceneUboMapped_[frame_], &uboData, sizeof(SceneUniformBuffer));

        // =====================================================================
        // PASS 1: DIRECTIONAL SHADOW DEPTH PASS
        // =====================================================================
        VkClearValue shadowClear{};
        shadowClear.depthStencil = {1.0f, 0};

        VkRenderPassBeginInfo srp{};
        srp.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        srp.renderPass = shadowRenderPass_;
        srp.framebuffer = shadowFramebuffer_;
        srp.renderArea = {{0, 0}, {kShadowMapDim, kShadowMapDim}};
        srp.clearValueCount = 1;
        srp.pClearValues = &shadowClear;

        vkCmdBeginRenderPass(cmd, &srp, VK_SUBPASS_CONTENTS_INLINE);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, shadowPipeline_);

        VkDeviceSize offset = 0;
        for (const auto& m : worldMeshes_) {
            if (m.isWater > 0.5f) continue; // Water does not cast opaque shadows
            Mat4 mvp = multiply(shadowViewProj, m.modelMatrix);
            vkCmdPushConstants(cmd, shadowPipelineLayout_, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(Mat4), &mvp);
            vkCmdBindVertexBuffers(cmd, 0, 1, &m.vb, &offset);
            vkCmdBindIndexBuffer(cmd, m.ib, 0, VK_INDEX_TYPE_UINT32);
            vkCmdDrawIndexed(cmd, m.indexCount, 1, 0, 0, 0);
        }

        if (hasGlbModel_ && glbVertexBuffer_ != VK_NULL_HANDLE) {
            Mat4 model = translate(0.0f, 0.0f, -8.0f);
            Mat4 mvp = multiply(shadowViewProj, model);
            vkCmdPushConstants(cmd, shadowPipelineLayout_, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(Mat4), &mvp);
            vkCmdBindVertexBuffers(cmd, 0, 1, &glbVertexBuffer_, &offset);
            vkCmdBindIndexBuffer(cmd, glbIndexBuffer_, 0, VK_INDEX_TYPE_UINT32);
            for (const auto& r : glbRanges_) {
                vkCmdDrawIndexed(cmd, r.indexCount, 1, r.firstIndex, 0, 0);
            }
        }
        vkCmdEndRenderPass(cmd);

        // =====================================================================
        // PASS 2: MAIN FORWARD+ LIGHTING & VOLUMETRIC SKY PASS
        // =====================================================================
        VkClearValue clears[2]{};
        if (sunElev > 0.0f) {
            clears[0].color = {{0.05f * sunElev, 0.12f * sunElev, 0.24f * sunElev, 1.0f}};
        } else {
            clears[0].color = {{0.003f, 0.005f, 0.01f, 1.0f}};
        }
        clears[1].depthStencil = {1.0f, 0};

        VkRenderPassBeginInfo rp{};
        rp.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        rp.renderPass = renderPass_;
        rp.framebuffer = framebuffers_[imageIndex];
        rp.renderArea = {{0, 0}, extent_};
        rp.clearValueCount = 2;
        rp.pClearValues = clears;

        vkCmdBeginRenderPass(cmd, &rp, VK_SUBPASS_CONTENTS_INLINE);

        // 1. Volumetric Sky & Atmosphere
        if (skyPipeline_ != VK_NULL_HANDLE) {
            SkyPushConstants skyPush{};
            skyPush.invViewProj = uboData.invViewProj;
            skyPush.cameraPos[0] = eye.x;
            skyPush.cameraPos[1] = eye.y;
            skyPush.cameraPos[2] = eye.z;
            skyPush.cameraPos[3] = t;
            skyPush.sunDir[0] = sunDir.x;
            skyPush.sunDir[1] = sunDir.y;
            skyPush.sunDir[2] = sunDir.z;
            skyPush.sunDir[3] = exposure_.load();
            skyPush.envParams[0] = fogDensity_.load();
            skyPush.envParams[1] = timeOfDay;
            skyPush.envParams[2] = 0.5f;
            skyPush.envParams[3] = 1.0f;

            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, skyPipeline_);
            vkCmdPushConstants(cmd, skyPipelineLayout_, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(SkyPushConstants), &skyPush);
            vkCmdDraw(cmd, 3, 1, 0, 0); // Fullscreen Quad
        }

        // 2. High-Poly PBR World Mesh Rendering with Soft Shadows
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pbrPipeline_);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pbrPipelineLayout_, 0, 1, &sceneDescSets_[frame_], 0, nullptr);

        ObjectPushConstants objPush{};
        for (const auto& m : worldMeshes_) {
            objPush.model = m.modelMatrix;
            std::copy(m.baseColor, m.baseColor + 4, objPush.baseColor);
            objPush.pbrParams[0] = m.metallic;
            objPush.pbrParams[1] = m.roughness;
            objPush.pbrParams[2] = 1.0f; // Ambient Occlusion default
            objPush.pbrParams[3] = m.isWater;

            vkCmdPushConstants(cmd, pbrPipelineLayout_, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(ObjectPushConstants), &objPush);
            vkCmdBindVertexBuffers(cmd, 0, 1, &m.vb, &offset);
            vkCmdBindIndexBuffer(cmd, m.ib, 0, VK_INDEX_TYPE_UINT32);
            vkCmdDrawIndexed(cmd, m.indexCount, 1, 0, 0, 0);
        }

        // 3. Dynamic GLB Mesh Rendering
        if (hasGlbModel_ && glbVertexBuffer_ != VK_NULL_HANDLE) {
            objPush.model = translate(0.0f, 0.0f, -8.0f);
            vkCmdBindVertexBuffers(cmd, 0, 1, &glbVertexBuffer_, &offset);
            vkCmdBindIndexBuffer(cmd, glbIndexBuffer_, 0, VK_INDEX_TYPE_UINT32);

            for (const auto& r : glbRanges_) {
                std::copy(r.baseColor, r.baseColor + 4, objPush.baseColor);
                objPush.pbrParams[0] = r.metallic;
                objPush.pbrParams[1] = r.roughness;
                objPush.pbrParams[2] = 1.0f;
                objPush.pbrParams[3] = 0.0f;

                vkCmdPushConstants(cmd, pbrPipelineLayout_, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(ObjectPushConstants), &objPush);
                vkCmdDrawIndexed(cmd, r.indexCount, 1, r.firstIndex, 0, 0);
            }
        }

        vkCmdEndRenderPass(cmd);
        check(vkEndCommandBuffer(cmd), "End command buffer");
    }

    void renderLoop() {
        try {
        auto lastTime = std::chrono::steady_clock::now();
        while (running_.load()) {
            if (resizeRequested_.exchange(false)) {
                std::lock_guard<std::mutex> lock(lifecycleMutex_);
                rebuild();
            }

            auto now = std::chrono::steady_clock::now();
            float dt = std::chrono::duration<float>(now - lastTime).count();
            lastTime = now;
            dt = std::clamp(dt, 0.001f, 0.05f);

            std::lock_guard<std::mutex> gpuLock(gpuMutex_);
            check(vkWaitForFences(device_, 1, &fences_[frame_], VK_TRUE, UINT64_MAX), "Fence wait");

            uint32_t imageIndex = 0;
            VkResult ac = vkAcquireNextImageKHR(device_, swapchain_, UINT64_MAX, imageAvailable_[frame_], VK_NULL_HANDLE, &imageIndex);
            if (ac == VK_ERROR_OUT_OF_DATE_KHR) { resizeRequested_.store(true); continue; }
            if (ac == VK_SUBOPTIMAL_KHR) resizeRequested_.store(true);
            else if (ac != VK_SUCCESS) check(ac, "vkAcquireNextImageKHR");

            check(vkResetFences(device_, 1, &fences_[frame_]), "vkResetFences");
            vkResetCommandBuffer(commandBuffers_[imageIndex], 0);
            record(imageIndex, dt);

            VkPipelineStageFlags wait = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
            VkSubmitInfo si{};
            si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            si.waitSemaphoreCount = 1;
            si.pWaitSemaphores = &imageAvailable_[frame_];
            si.pWaitDstStageMask = &wait;
            si.commandBufferCount = 1;
            si.pCommandBuffers = &commandBuffers_[imageIndex];
            si.signalSemaphoreCount = 1;
            si.pSignalSemaphores = &renderFinished_[frame_];

            check(vkQueueSubmit(graphicsQueue_, 1, &si, fences_[frame_]), "Queue submit");

            VkPresentInfoKHR pi{};
            pi.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
            pi.waitSemaphoreCount = 1;
            pi.pWaitSemaphores = &renderFinished_[frame_];
            pi.swapchainCount = 1;
            pi.pSwapchains = &swapchain_;
            pi.pImageIndices = &imageIndex;

            VkResult pr = vkQueuePresentKHR(presentQueue_, &pi);
            if (pr == VK_ERROR_OUT_OF_DATE_KHR || pr == VK_SUBOPTIMAL_KHR) {
                resizeRequested_.store(true);
            } else if (pr != VK_SUCCESS) {
                check(pr, "vkQueuePresentKHR");
            }

            frame_ = (frame_ + 1) % kFrames;
        }
        } catch (const std::exception& e) {
            running_.store(false);
            setStatus(std::string("Vulkan render loop stopped safely: ") + e.what());
            LOGE("%s", status_.c_str());
        } catch (...) {
            running_.store(false);
            setStatus("Vulkan render loop stopped safely: unknown native exception");
            LOGE("%s", status_.c_str());
        }
    }

    void rebuild() {
        if (!device_) return;
        vkDeviceWaitIdle(device_);
        destroySwapchain();
        createSwapchain();
        chooseDepthFormat();
        createRenderPass();
        createDepthResources();
        createFramebuffers();
        createCommands();
        createPBRPipeline();
        createSkyPipeline();
    }

    void destroySwapchain() {
        if (!device_) return;
        if (pbrPipeline_) vkDestroyPipeline(device_, pbrPipeline_, nullptr); pbrPipeline_ = VK_NULL_HANDLE;
        if (skyPipeline_) vkDestroyPipeline(device_, skyPipeline_, nullptr); skyPipeline_ = VK_NULL_HANDLE;
        if (skyPipelineLayout_) vkDestroyPipelineLayout(device_, skyPipelineLayout_, nullptr); skyPipelineLayout_ = VK_NULL_HANDLE;
        if (pbrPipelineLayout_) vkDestroyPipelineLayout(device_, pbrPipelineLayout_, nullptr); pbrPipelineLayout_ = VK_NULL_HANDLE;

        for (auto f : framebuffers_) if (f) vkDestroyFramebuffer(device_, f, nullptr);
        framebuffers_.clear();
        if (commandPool_) vkDestroyCommandPool(device_, commandPool_, nullptr); commandPool_ = VK_NULL_HANDLE;
        commandBuffers_.clear();
        if (renderPass_) vkDestroyRenderPass(device_, renderPass_, nullptr); renderPass_ = VK_NULL_HANDLE;

        for (auto v : depthViews_) if (v) vkDestroyImageView(device_, v, nullptr); depthViews_.clear();
        for (auto im : depthImages_) if (im) vkDestroyImage(device_, im, nullptr); depthImages_.clear();
        for (auto m : depthMemory_) if (m) vkFreeMemory(device_, m, nullptr); depthMemory_.clear();
        for (auto v : views_) if (v) vkDestroyImageView(device_, v, nullptr); views_.clear();
        images_.clear();
        if (swapchain_) vkDestroySwapchainKHR(device_, swapchain_, nullptr); swapchain_ = VK_NULL_HANDLE;
    }

    void cleanup() {
        if (device_) vkDeviceWaitIdle(device_);
        destroySwapchain();
        if (device_) {
            for (auto& m : worldMeshes_) {
                if (m.vb) vkDestroyBuffer(device_, m.vb, nullptr);
                if (m.ib) vkDestroyBuffer(device_, m.ib, nullptr);
                if (m.vMem) vkFreeMemory(device_, m.vMem, nullptr);
                if (m.iMem) vkFreeMemory(device_, m.iMem, nullptr);
            }
            worldMeshes_.clear();

            if (glbVertexBuffer_) vkDestroyBuffer(device_, glbVertexBuffer_, nullptr);
            if (glbIndexBuffer_) vkDestroyBuffer(device_, glbIndexBuffer_, nullptr);
            if (glbVertexMemory_) vkFreeMemory(device_, glbVertexMemory_, nullptr);
            if (glbIndexMemory_) vkFreeMemory(device_, glbIndexMemory_, nullptr);

            if (shadowPipeline_) vkDestroyPipeline(device_, shadowPipeline_, nullptr);
            if (shadowPipelineLayout_) vkDestroyPipelineLayout(device_, shadowPipelineLayout_, nullptr);
            if (shadowFramebuffer_) vkDestroyFramebuffer(device_, shadowFramebuffer_, nullptr);
            if (shadowRenderPass_) vkDestroyRenderPass(device_, shadowRenderPass_, nullptr);
            if (shadowSampler_) vkDestroySampler(device_, shadowSampler_, nullptr);
            if (shadowView_) vkDestroyImageView(device_, shadowView_, nullptr);
            if (shadowImage_) vkDestroyImage(device_, shadowImage_, nullptr);
            if (shadowMemory_) vkFreeMemory(device_, shadowMemory_, nullptr);

            for (size_t i = 0; i < kFrames; ++i) {
                if (sceneUboMapped_[i]) vkUnmapMemory(device_, sceneUboMemory_[i]);
                if (sceneUboBuffers_[i]) vkDestroyBuffer(device_, sceneUboBuffers_[i], nullptr);
                if (sceneUboMemory_[i]) vkFreeMemory(device_, sceneUboMemory_[i], nullptr);
            }
            if (descPool_) vkDestroyDescriptorPool(device_, descPool_, nullptr);
            if (sceneDescLayout_) vkDestroyDescriptorSetLayout(device_, sceneDescLayout_, nullptr);

            for (size_t i = 0; i < kFrames; ++i) {
                if (imageAvailable_[i]) vkDestroySemaphore(device_, imageAvailable_[i], nullptr);
                if (renderFinished_[i]) vkDestroySemaphore(device_, renderFinished_[i], nullptr);
                if (fences_[i]) vkDestroyFence(device_, fences_[i], nullptr);
            }
            vkDestroyDevice(device_, nullptr);
            device_ = VK_NULL_HANDLE;
        }
        if (instance_ && surface_) {
            vkDestroySurfaceKHR(instance_, surface_, nullptr);
            surface_ = VK_NULL_HANDLE;
        }
        if (instance_) {
            vkDestroyInstance(instance_, nullptr);
            instance_ = VK_NULL_HANDLE;
        }
    }
};

static std::mutex gMutex;
static VulkanRenderer* gRenderer = nullptr;

} // namespace neo3d

// -----------------------------------------------------------------------------
// JNI Method Bindings (Directly compatible sa Java Activity mo)
// -----------------------------------------------------------------------------
extern "C" JNIEXPORT void JNICALL Java_com_neo3d_engine_MainActivity_nativeStart(JNIEnv* env, jobject, jobject surface) {
    std::lock_guard<std::mutex> lock(neo3d::gMutex);
    if (neo3d::gRenderer) {
        neo3d::gRenderer->stop();
        delete neo3d::gRenderer;
        neo3d::gRenderer = nullptr;
    }
    ANativeWindow* w = ANativeWindow_fromSurface(env, surface);
    if (!w) return;
    neo3d::gRenderer = new neo3d::VulkanRenderer(w);
    ANativeWindow_release(w);
    neo3d::gRenderer->start();
}

extern "C" JNIEXPORT void JNICALL Java_com_neo3d_engine_MainActivity_nativeResize(JNIEnv*, jobject, jint w, jint h) {
    std::lock_guard<std::mutex> lock(neo3d::gMutex);
    if (neo3d::gRenderer) neo3d::gRenderer->resize(w, h);
}

extern "C" JNIEXPORT void JNICALL Java_com_neo3d_engine_MainActivity_nativeStop(JNIEnv*, jobject) {
    std::lock_guard<std::mutex> lock(neo3d::gMutex);
    if (neo3d::gRenderer) {
        neo3d::gRenderer->stop();
        delete neo3d::gRenderer;
        neo3d::gRenderer = nullptr;
    }
}

extern "C" JNIEXPORT jstring JNICALL Java_com_neo3d_engine_MainActivity_nativeStatus(JNIEnv* env, jobject) {
    std::lock_guard<std::mutex> lock(neo3d::gMutex);
    std::string s = neo3d::gRenderer ? neo3d::gRenderer->status() : "Engine Stopped";
    return env->NewStringUTF(s.c_str());
}

extern "C" JNIEXPORT jstring JNICALL Java_com_neo3d_engine_MainActivity_nativeLoadGlb(JNIEnv* env, jobject, jbyteArray data) {
    if (!data) return env->NewStringUTF("GLB failed: null data");
    jsize len = env->GetArrayLength(data);
    std::vector<uint8_t> bytes(static_cast<size_t>(len));
    env->GetByteArrayRegion(data, 0, len, reinterpret_cast<jbyte*>(bytes.data()));

    std::lock_guard<std::mutex> lock(neo3d::gMutex);
    if (!neo3d::gRenderer) return env->NewStringUTF("GLB failed: start engine first");
    std::string res = neo3d::gRenderer->loadGlb(bytes);
    return env->NewStringUTF(res.c_str());
}

extern "C" JNIEXPORT void JNICALL Java_com_neo3d_engine_MainActivity_nativeLook(JNIEnv*, jobject, jfloat dx, jfloat dy) {
    std::lock_guard<std::mutex> lock(neo3d::gMutex);
    if (neo3d::gRenderer) neo3d::gRenderer->look(dx, dy);
}

extern "C" JNIEXPORT void JNICALL Java_com_neo3d_engine_MainActivity_nativeJoystickMove(JNIEnv*, jobject, jfloat ix, jfloat iy) {
    std::lock_guard<std::mutex> lock(neo3d::gMutex);
    if (neo3d::gRenderer) neo3d::gRenderer->setJoystickInput(ix, iy);
}

extern "C" JNIEXPORT void JNICALL Java_com_neo3d_engine_MainActivity_nativeJump(JNIEnv*, jobject) {
    std::lock_guard<std::mutex> lock(neo3d::gMutex);
    if (neo3d::gRenderer) neo3d::gRenderer->triggerJump();
}

extern "C" JNIEXPORT void JNICALL Java_com_neo3d_engine_MainActivity_nativeUpdateSettings(JNIEnv*, jobject, jfloat fog, jfloat time, jfloat exp) {
    std::lock_guard<std::mutex> lock(neo3d::gMutex);
    if (neo3d::gRenderer) neo3d::gRenderer->updateSettings(fog, time, exp);
}

extern "C" JNIEXPORT void JNICALL Java_com_neo3d_engine_MainActivity_nativeResetView(JNIEnv*, jobject) {
    std::lock_guard<std::mutex> lock(neo3d::gMutex);
    if (neo3d::gRenderer) neo3d::gRenderer->resetView();
}
