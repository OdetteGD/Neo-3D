#include <jni.h>
#include <android/native_window.h>
#include <android/native_window_jni.h>
#include <android/log.h>
#define VK_USE_PLATFORM_ANDROID_KHR
#include <vulkan/vulkan.h>
#include "shader_blobs.hpp"
#include "assets/glb_reader.hpp"

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
// Math & Geometric Utilities
// -----------------------------------------------------------------------------
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

struct Vertex {
    float position[3];
    float normal[3];
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
    const float f = 1.0f / std::tan(0.78539816339f * 0.5f);
    const float nearP = 0.05f, farP = 250.0f;
    Mat4 m{};
    m.v[0] = f / std::max(aspect, 0.01f);
    m.v[5] = -f; // Vulkan inverted Y
    m.v[10] = farP / (nearP - farP);
    m.v[11] = -1.0f;
    m.v[14] = (farP * nearP) / (nearP - farP);
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
    Vec3 s = {f.y * up.z - f.z * up.y, f.z * up.x - f.x * up.z, f.x * up.y - f.y * up.x};
    s = s.normalized();
    Vec3 u = {s.y * f.z - s.z * f.y, s.z * f.x - s.x * f.z, s.x * f.y - s.y * f.x};

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
// Physics Character Controller & Collision Entities
// -----------------------------------------------------------------------------
struct PhysicsBox {
    AABB box;
    float color[4];
    float metallic;
    float roughness;
};

class CharacterController {
public:
    Vec3 position{0.0f, 2.0f, 5.0f};
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
            velocity.y = 7.5f;
            isGrounded = false;
        }
    }

    void update(float dt, float inputX, float inputY, float yaw, const std::vector<PhysicsBox>& colliders) {
        float sinY = std::sin(yaw);
        float cosY = std::cos(yaw);
        Vec3 forward = {sinY, 0.0f, -cosY};
        Vec3 right = {cosY, 0.0f, sinY};

        float speed = 6.8f;
        Vec3 targetMove = (forward * inputY + right * inputX) * speed;

        velocity.x = targetMove.x;
        velocity.z = targetMove.z;

        const float kGravity = -18.5f;
        velocity.y += kGravity * dt;
        if (velocity.y < -30.0f) velocity.y = -30.0f;

        Vec3 stepX = position;
        stepX.x += velocity.x * dt;
        AABB boxX = getAABB(stepX);
        bool colX = false;
        for (const auto& c : colliders) {
            if (boxX.intersects(c.box)) { colX = true; break; }
        }
        if (!colX) position.x = stepX.x;

        Vec3 stepZ = position;
        stepZ.z += velocity.z * dt;
        AABB boxZ = getAABB(stepZ);
        bool colZ = false;
        for (const auto& c : colliders) {
            if (boxZ.intersects(c.box)) { colZ = true; break; }
        }
        if (!colZ) position.z = stepZ.z;

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
            for (const auto& c : colliders) {
                if (boxY.intersects(c.box)) {
                    if (velocity.y < 0.0f) {
                        position.y = c.box.max.y;
                        velocity.y = 0.0f;
                        isGrounded = true;
                    } else if (velocity.y > 0.0f) {
                        position.y = c.box.min.y - height;
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
// Vulkan High-Performance Renderer Engine
// -----------------------------------------------------------------------------
class VulkanRenderer {
public:
    explicit VulkanRenderer(ANativeWindow* w) : window_(w) {
        if (window_) ANativeWindow_acquire(window_);
        buildWorldChunks();
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
            status_ = std::string("Vulkan init failed: ") + e.what();
            LOGE("%s", status_.c_str());
            cleanup();
        }
    }

    void resize(int, int) { resizeRequested_.store(true); }

    void stop() {
        running_.store(false);
        if (thread_.joinable()) thread_.join();
        cleanup();
    }

    std::string status() const { return status_; }

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
        player_.position = {0.0f, 2.0f, 6.0f};
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
            if (vertexTotal == 0 || indexTotal == 0) throw std::runtime_error("Empty GLB mesh");

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
            const float scaleFactor = 2.5f / std::max(span, 1e-4f);

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
                    vertices.push_back(out);
                }
                for (auto idx : p.indices) indices.push_back(base + idx);
                ranges.push_back(range);
            }

            std::lock_guard<std::mutex> lock(gpuMutex_);
            if (!device_) throw std::runtime_error("Device not ready");
            vkDeviceWaitIdle(device_);

            if (glbVertexBuffer_) vkDestroyBuffer(device_, glbVertexBuffer_, nullptr);
            if (glbIndexBuffer_) vkDestroyBuffer(device_, glbIndexBuffer_, nullptr);
            if (glbVertexMemory_) vkFreeMemory(device_, glbVertexMemory_, nullptr);
            if (glbIndexMemory_) vkFreeMemory(device_, glbIndexMemory_, nullptr);

            createBuffer(vertices.size() * sizeof(Vertex), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, glbVertexBuffer_, glbVertexMemory_, vertices.data());
            createBuffer(indices.size() * sizeof(uint32_t), VK_BUFFER_USAGE_INDEX_BUFFER_BIT, glbIndexBuffer_, glbIndexMemory_, indices.data());
            glbRanges_ = std::move(ranges);
            hasGlbModel_ = true;

            status_ = "GLB Imported (" + std::to_string(vertexTotal) + " verts)";
            return status_;
        } catch (const std::exception& e) {
            status_ = std::string("GLB failed: ") + e.what();
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
    VkCommandPool commandPool_ = VK_NULL_HANDLE;
    std::vector<VkCommandBuffer> commandBuffers_;

    VkPipelineLayout pipelineLayout_ = VK_NULL_HANDLE;
    VkPipeline pipeline_ = VK_NULL_HANDLE;
    VkPipelineLayout skyPipelineLayout_ = VK_NULL_HANDLE;
    VkPipeline skyPipeline_ = VK_NULL_HANDLE;

    // Static Cube Geometry
    VkBuffer cubeVertexBuffer_ = VK_NULL_HANDLE, cubeIndexBuffer_ = VK_NULL_HANDLE;
    VkDeviceMemory cubeVertexMemory_ = VK_NULL_HANDLE, cubeIndexMemory_ = VK_NULL_HANDLE;
    uint32_t cubeIndexCount_ = 0;

    // Dynamic GLB Geometry
    VkBuffer glbVertexBuffer_ = VK_NULL_HANDLE, glbIndexBuffer_ = VK_NULL_HANDLE;
    VkDeviceMemory glbVertexMemory_ = VK_NULL_HANDLE, glbIndexMemory_ = VK_NULL_HANDLE;
    bool hasGlbModel_ = false;

    struct DrawRange {
        uint32_t firstIndex = 0, indexCount = 0;
        float baseColor[4]{1.0f, 1.0f, 1.0f, 1.0f};
        float metallic = 0.5f, roughness = 0.5f;
    };
    std::vector<DrawRange> glbRanges_;

    static constexpr size_t kFrames = 2;
    std::array<VkSemaphore, kFrames> imageAvailable_{};
    std::array<VkSemaphore, kFrames> renderFinished_{};
    std::array<VkFence, kFrames> fences_{};
    size_t frame_ = 0;

    std::atomic<bool> running_{false}, resizeRequested_{false}, jumpRequested_{false};
    std::mutex gpuMutex_, physicsMutex_, lifecycleMutex_;

    // FPS Camera & Physics Controllers
    CharacterController player_;
    std::vector<PhysicsBox> worldBoxes_;
    std::atomic<float> yaw_{0.0f}, pitch_{-0.1f};
    std::atomic<float> inputX_{0.0f}, inputY_{0.0f};

    // Environmental Engine Parameters
    std::atomic<float> fogDensity_{0.045f};
    std::atomic<float> timeOfDay_{14.0f};
    std::atomic<float> exposure_{1.0f};

    std::thread thread_;
    std::chrono::steady_clock::time_point started_{};
    std::string deviceName_ = "GPU", status_ = "Init";

    struct MeshPushConstants {
        Mat4 mvp;
        Mat4 model;
        float baseColor[4];
        float material[4];
        float cameraPos[4];
        float sunDir[4];
        float envParams[4];
    };

    struct SkyPushConstants {
        Mat4 invViewProj;
        float cameraPos[4];
        float sunDir[4];
        float envParams[4];
    };

    void buildWorldChunks() {
        worldBoxes_.clear();
        worldBoxes_.push_back({{{-60.0f, -1.0f, -60.0f}, {60.0f, 0.0f, 60.0f}}, {0.22f, 0.28f, 0.24f, 1.0f}, 0.05f, 0.90f});

        for (int i = 0; i < 7; ++i) {
            float h = (i + 1) * 0.45f;
            float z = -2.0f - (i * 1.2f);
            worldBoxes_.push_back({{{-1.5f, 0.0f, z - 0.6f}, {1.5f, h, z + 0.6f}}, {0.65f, 0.58f, 0.48f, 1.0f}, 0.15f, 0.75f});
        }

        worldBoxes_.push_back({{{-6.0f, 3.15f, -18.0f}, {6.0f, 3.55f, -8.0f}}, {0.35f, 0.45f, 0.60f, 1.0f}, 0.40f, 0.35f});
        worldBoxes_.push_back({{{-7.0f, 0.0f, -6.0f}, {-5.0f, 7.0f, -4.0f}}, {0.95f, 0.85f, 0.30f, 1.0f}, 0.95f, 0.15f});
        worldBoxes_.push_back({{{5.0f, 0.0f, -6.0f}, {7.0f, 7.0f, -4.0f}}, {0.92f, 0.92f, 0.95f, 1.0f}, 0.98f, 0.08f});
    }

    void initialize() {
        VkApplicationInfo app{};
        app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        app.pApplicationName = "Neo-3D";
        app.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
        app.pEngineName = "Neo-3D Engine";
        app.engineVersion = VK_MAKE_VERSION(1, 0, 0);
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
        createCubeGeometry();
        createSwapchain();
        chooseDepthFormat();
        createRenderPass();
        createDepthResources();
        createFramebuffers();
        createCommands();
        createGraphicsPipeline();
        createSkyPipeline();
        createSync();

        status_ = "Engine Online | " + deviceName_ + " | PBR & Volumetrics Active";
    }

    void selectDevice() {
        uint32_t count = 0;
        check(vkEnumeratePhysicalDevices(instance_, &count, nullptr), "vkEnumeratePhysicalDevices count");
        std::vector<VkPhysicalDevice> ds(count);
        check(vkEnumeratePhysicalDevices(instance_, &count, ds.data()), "vkEnumeratePhysicalDevices");

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
        throw std::runtime_error("No physical device supports graphics and present");
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
        VkDeviceCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        ci.queueCreateInfoCount = static_cast<uint32_t>(qs.size());
        ci.pQueueCreateInfos = qs.data();
        ci.enabledExtensionCount = 1;
        ci.ppEnabledExtensionNames = &ext;
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
        throw std::runtime_error("No matching memory type found");
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

    void createCubeGeometry() {
        const std::array<Vertex, 24> v = {{
            {{-0.5f, -0.5f,  0.5f}, { 0,  0,  1}}, {{ 0.5f, -0.5f,  0.5f}, { 0,  0,  1}}, {{ 0.5f,  0.5f,  0.5f}, { 0,  0,  1}}, {{-0.5f,  0.5f,  0.5f}, { 0,  0,  1}},
            {{ 0.5f, -0.5f, -0.5f}, { 0,  0, -1}}, {{-0.5f, -0.5f, -0.5f}, { 0,  0, -1}}, {{-0.5f,  0.5f, -0.5f}, { 0,  0, -1}}, {{ 0.5f,  0.5f, -0.5f}, { 0,  0, -1}},
            {{-0.5f, -0.5f, -0.5f}, {-1,  0,  0}}, {{-0.5f, -0.5f,  0.5f}, {-1,  0,  0}}, {{-0.5f,  0.5f,  0.5f}, {-1,  0,  0}}, {{-0.5f,  0.5f, -0.5f}, {-1,  0,  0}},
            {{ 0.5f, -0.5f,  0.5f}, { 1,  0,  0}}, {{ 0.5f, -0.5f, -0.5f}, { 1,  0,  0}}, {{ 0.5f,  0.5f, -0.5f}, { 1,  0,  0}}, {{ 0.5f,  0.5f,  0.5f}, { 1,  0,  0}},
            {{-0.5f,  0.5f,  0.5f}, { 0,  1,  0}}, {{ 0.5f,  0.5f,  0.5f}, { 0,  1,  0}}, {{ 0.5f,  0.5f, -0.5f}, { 0,  1,  0}}, {{-0.5f,  0.5f, -0.5f}, { 0,  1,  0}},
            {{-0.5f, -0.5f, -0.5f}, { 0, -1,  0}}, {{ 0.5f, -0.5f, -0.5f}, { 0, -1,  0}}, {{ 0.5f, -0.5f,  0.5f}, { 0, -1,  0}}, {{-0.5f, -0.5f,  0.5f}, { 0, -1,  0}}
        }};
        const std::array<uint32_t, 36> idx = {{
             0,  1,  2,  2,  3,  0,   4,  5,  6,  6,  7,  4,
             8,  9, 10, 10, 11,  8,  12, 13, 14, 14, 15, 12,
            16, 17, 18, 18, 19, 16,  20, 21, 22, 22, 23, 20
        }};
        cubeIndexCount_ = static_cast<uint32_t>(idx.size());
        createBuffer(sizeof(v), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, cubeVertexBuffer_, cubeVertexMemory_, v.data());
        createBuffer(sizeof(idx), VK_BUFFER_USAGE_INDEX_BUFFER_BIT, cubeIndexBuffer_, cubeIndexMemory_, idx.data());
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

    void chooseDepthFormat() {
        for (VkFormat f : {VK_FORMAT_D32_SFLOAT, VK_FORMAT_D24_UNORM_S8_UINT, VK_FORMAT_D16_UNORM}) {
            VkFormatProperties p{};
            vkGetPhysicalDeviceFormatProperties(physical_, f, &p);
            if (p.optimalTilingFeatures & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT) {
                depthFormat_ = f; return;
            }
        }
        throw std::runtime_error("No suitable depth format");
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

    void createCommands() {
        VkCommandPoolCreateInfo pi{};
        pi.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        pi.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        pi.queueFamilyIndex = graphicsFamily_;
        check(vkCreateCommandPool(device_, &pi, nullptr, &commandPool_), "vkCreateCommandPool");

        commandBuffers_.resize(framebuffers_.size());
        VkCommandBufferAllocateInfo ai{};
        ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        ai.commandPool = commandPool_;
        ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        ai.commandBufferCount = static_cast<uint32_t>(commandBuffers_.size());
        check(vkAllocateCommandBuffers(device_, &ai, commandBuffers_.data()), "vkAllocateCommandBuffers");
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

    void createGraphicsPipeline() {
        VkShaderModule v = createShader(kMeshVert, kMeshVertSize);
        VkShaderModule f = createShader(kMeshFrag, kMeshFragSize);

        VkPushConstantRange range{};
        range.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
        range.offset = 0;
        range.size = sizeof(MeshPushConstants);

        VkPipelineLayoutCreateInfo li{};
        li.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        li.pushConstantRangeCount = 1;
        li.pPushConstantRanges = &range;
        check(vkCreatePipelineLayout(device_, &li, nullptr, &pipelineLayout_), "vkCreatePipelineLayout");

        VkPipelineShaderStageCreateInfo stages[2]{};
        stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
        stages[0].module = v;
        stages[0].pName = "main";
        stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        stages[1].module = f;
        stages[1].pName = "main";

        VkVertexInputBindingDescription b{};
        b.binding = 0;
        b.stride = sizeof(Vertex);
        b.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

        VkVertexInputAttributeDescription a[2]{};
        a[0].location = 0;
        a[0].binding = 0;
        a[0].format = VK_FORMAT_R32G32B32_SFLOAT;
        a[0].offset = offsetof(Vertex, position);
        a[1].location = 1;
        a[1].binding = 0;
        a[1].format = VK_FORMAT_R32G32B32_SFLOAT;
        a[1].offset = offsetof(Vertex, normal);

        VkPipelineVertexInputStateCreateInfo vi{};
        vi.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        vi.vertexBindingDescriptionCount = 1;
        vi.pVertexBindingDescriptions = &b;
        vi.vertexAttributeDescriptionCount = 2;
        vi.pVertexAttributeDescriptions = a;

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
        rs.lineWidth = 1.0f;

        VkPipelineMultisampleStateCreateInfo ms{};
        ms.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

        VkPipelineDepthStencilStateCreateInfo ds{};
        ds.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
        ds.depthTestEnable = VK_TRUE;
        ds.depthWriteEnable = VK_TRUE;
        ds.depthCompareOp = VK_COMPARE_OP_LESS;

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
        pi.layout = pipelineLayout_;
        pi.renderPass = renderPass_;

        check(vkCreateGraphicsPipelines(device_, VK_NULL_HANDLE, 1, &pi, nullptr, &pipeline_), "vkCreateGraphicsPipelines");
        vkDestroyShaderModule(device_, v, nullptr);
        vkDestroyShaderModule(device_, f, nullptr);
    }

    void createSkyPipeline() {
        VkShaderModule v = createShader(kSkyVert, kSkyVertSize);
        VkShaderModule f = createShader(kSkyFrag, kSkyFragSize);

        VkPushConstantRange range{};
        range.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
        range.offset = 0;
        range.size = sizeof(SkyPushConstants);

        VkPipelineLayoutCreateInfo li{};
        li.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        li.pushConstantRangeCount = 1;
        li.pPushConstantRanges = &range;
        check(vkCreatePipelineLayout(device_, &li, nullptr, &skyPipelineLayout_), "vkCreatePipelineLayout sky");

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

        check(vkCreateGraphicsPipelines(device_, VK_NULL_HANDLE, 1, &pi, nullptr, &skyPipeline_), "vkCreateGraphicsPipelines sky");
        vkDestroyShaderModule(device_, v, nullptr);
        vkDestroyShaderModule(device_, f, nullptr);
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

    void record(uint32_t i, float dt) {
        VkCommandBufferBeginInfo bi{};
        bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        check(vkBeginCommandBuffer(commandBuffers_[i], &bi), "vkBeginCommandBuffer");

        float timeOfDay = timeOfDay_.load();
        float sunAngle = (timeOfDay / 24.0f) * 6.2831853f - 1.5707963f;
        float sunElev = std::sin(sunAngle);

        VkClearValue clears[2]{};
        if (sunElev > 0.0f) {
            clears[0].color = {{0.08f * sunElev, 0.16f * sunElev, 0.28f * sunElev, 1.0f}};
        } else {
            clears[0].color = {{0.005f, 0.008f, 0.015f, 1.0f}};
        }
        clears[1].depthStencil = {1.0f, 0};

        VkRenderPassBeginInfo rp{};
        rp.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        rp.renderPass = renderPass_;
        rp.framebuffer = framebuffers_[i];
        rp.renderArea = {{0, 0}, extent_};
        rp.clearValueCount = 2;
        rp.pClearValues = clears;

        vkCmdBeginRenderPass(commandBuffers_[i], &rp, VK_SUBPASS_CONTENTS_INLINE);

        float t = std::chrono::duration<float>(std::chrono::steady_clock::now() - started_).count();

        // 1. Physics Step
        {
            std::lock_guard<std::mutex> lock(physicsMutex_);
            if (jumpRequested_.exchange(false)) {
                player_.jump();
            }
            player_.update(dt, inputX_.load(), inputY_.load(), yaw_.load(), worldBoxes_);
        }

        // 2. Camera View & Projection
        Vec3 eye = player_.getEyePosition();
        float yaw = yaw_.load();
        float pitch = pitch_.load();

        Vec3 forward = {
            std::sin(yaw) * std::cos(pitch),
            -std::sin(pitch),
            -std::cos(yaw) * std::cos(pitch)
        };
        Vec3 target = eye + forward;

        Mat4 view = lookAt(eye, target, {0.0f, 1.0f, 0.0f});
        Mat4 proj = perspective(static_cast<float>(extent_.width) / static_cast<float>(std::max(1u, extent_.height)));
        Mat4 viewProj = multiply(proj, view);

        float sunDir[4] = {
            std::cos(sunAngle),
            sunElev,
            0.35f,
            exposure_.load()
        };

        // 3. Volumetric Sky Pass
        if (skyPipeline_ != VK_NULL_HANDLE) {
            SkyPushConstants skyPush{};
            skyPush.invViewProj = inverseMat4(viewProj);
            skyPush.cameraPos[0] = eye.x;
            skyPush.cameraPos[1] = eye.y;
            skyPush.cameraPos[2] = eye.z;
            skyPush.cameraPos[3] = t;
            std::copy(sunDir, sunDir + 4, skyPush.sunDir);
            skyPush.envParams[0] = fogDensity_.load();
            skyPush.envParams[1] = timeOfDay;
            skyPush.envParams[2] = 0.52f;
            skyPush.envParams[3] = 1.0f;

            vkCmdBindPipeline(commandBuffers_[i], VK_PIPELINE_BIND_POINT_GRAPHICS, skyPipeline_);
            vkCmdPushConstants(
                commandBuffers_[i],
                skyPipelineLayout_,
                VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                0,
                sizeof(SkyPushConstants),
                &skyPush
            );
            vkCmdDraw(commandBuffers_[i], 3, 1, 0, 0);
        }

        // 4. PBR World Chunks & Geometry Pass
        vkCmdBindPipeline(commandBuffers_[i], VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_);

        VkDeviceSize offset = 0;
        vkCmdBindVertexBuffers(commandBuffers_[i], 0, 1, &cubeVertexBuffer_, &offset);
        vkCmdBindIndexBuffer(commandBuffers_[i], cubeIndexBuffer_, 0, VK_INDEX_TYPE_UINT32);

        MeshPushConstants push{};
        push.cameraPos[0] = eye.x;
        push.cameraPos[1] = eye.y;
        push.cameraPos[2] = eye.z;
        push.cameraPos[3] = t;
        std::copy(sunDir, sunDir + 4, push.sunDir);
        push.envParams[0] = fogDensity_.load();
        push.envParams[1] = timeOfDay;
        push.envParams[2] = 0.0f;
        push.envParams[3] = 0.0f;

        for (const auto& b : worldBoxes_) {
            Vec3 size = b.box.max - b.box.min;
            Vec3 center = (b.box.min + b.box.max) * 0.5f;

            Mat4 model = multiply(translate(center.x, center.y, center.z), scale(size.x, size.y, size.z));
            push.model = model;
            push.mvp = multiply(viewProj, model);

            std::copy(b.color, b.color + 4, push.baseColor);
            push.material[0] = b.metallic;
            push.material[1] = b.roughness;
            push.material[2] = 1.0f;
            push.material[3] = 0.0f;

            vkCmdPushConstants(
                commandBuffers_[i],
                pipelineLayout_,
                VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                0,
                sizeof(MeshPushConstants),
                &push
            );
            vkCmdDrawIndexed(commandBuffers_[i], cubeIndexCount_, 1, 0, 0, 0);
        }

        // 5. Dynamic GLB Mesh Rendering
        if (hasGlbModel_ && glbVertexBuffer_ != VK_NULL_HANDLE) {
            vkCmdBindVertexBuffers(commandBuffers_[i], 0, 1, &glbVertexBuffer_, &offset);
            vkCmdBindIndexBuffer(commandBuffers_[i], glbIndexBuffer_, 0, VK_INDEX_TYPE_UINT32);

            Mat4 model = translate(0.0f, 0.0f, -8.0f);
            push.model = model;
            push.mvp = multiply(viewProj, model);

            for (const auto& r : glbRanges_) {
                std::copy(r.baseColor, r.baseColor + 4, push.baseColor);
                push.material[0] = r.metallic;
                push.material[1] = r.roughness;
                push.material[2] = 1.0f;
                push.material[3] = 0.0f;

                vkCmdPushConstants(
                    commandBuffers_[i],
                    pipelineLayout_,
                    VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                    0,
                    sizeof(MeshPushConstants),
                    &push
                );
                vkCmdDrawIndexed(commandBuffers_[i], r.indexCount, 1, r.firstIndex, 0, 0);
            }
        }

        vkCmdEndRenderPass(commandBuffers_[i]);
        check(vkEndCommandBuffer(commandBuffers_[i]), "vkEndCommandBuffer");
    }

    void renderLoop() {
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
            check(vkWaitForFences(device_, 1, &fences_[frame_], VK_TRUE, UINT64_MAX), "fence wait");

            uint32_t imageIndex = 0;
            VkResult ac = vkAcquireNextImageKHR(device_, swapchain_, UINT64_MAX, imageAvailable_[frame_], VK_NULL_HANDLE, &imageIndex);
            if (ac == VK_ERROR_OUT_OF_DATE_KHR) { resizeRequested_.store(true); continue; }

            vkResetFences(device_, 1, &fences_[frame_]);
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

            check(vkQueueSubmit(graphicsQueue_, 1, &si, fences_[frame_]), "queue submit");

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
            }

            frame_ = (frame_ + 1) % kFrames;
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
        createGraphicsPipeline();
        createSkyPipeline();
    }

    void destroySwapchain() {
        if (!device_) return;
        if (pipeline_) vkDestroyPipeline(device_, pipeline_, nullptr); pipeline_ = VK_NULL_HANDLE;
        if (skyPipeline_) vkDestroyPipeline(device_, skyPipeline_, nullptr); skyPipeline_ = VK_NULL_HANDLE;
        if (skyPipelineLayout_) vkDestroyPipelineLayout(device_, skyPipelineLayout_, nullptr); skyPipelineLayout_ = VK_NULL_HANDLE;
        if (pipelineLayout_) vkDestroyPipelineLayout(device_, pipelineLayout_, nullptr); pipelineLayout_ = VK_NULL_HANDLE;

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
            if (cubeVertexBuffer_) vkDestroyBuffer(device_, cubeVertexBuffer_, nullptr);
            if (cubeIndexBuffer_) vkDestroyBuffer(device_, cubeIndexBuffer_, nullptr);
            if (cubeVertexMemory_) vkFreeMemory(device_, cubeVertexMemory_, nullptr);
            if (cubeIndexMemory_) vkFreeMemory(device_, cubeIndexMemory_, nullptr);

            if (glbVertexBuffer_) vkDestroyBuffer(device_, glbVertexBuffer_, nullptr);
            if (glbIndexBuffer_) vkDestroyBuffer(device_, glbIndexBuffer_, nullptr);
            if (glbVertexMemory_) vkFreeMemory(device_, glbVertexMemory_, nullptr);
            if (glbIndexMemory_) vkFreeMemory(device_, glbIndexMemory_, nullptr);

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
// JNI Method Bindings
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
    std::string s = neo3d::gRenderer ? neo3d::gRenderer->status() : "Stopped";
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
