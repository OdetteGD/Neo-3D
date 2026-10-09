#include <jni.h>
#include <android/native_window.h>
#include <android/native_window_jni.h>
#include <android/log.h>
#define VK_USE_PLATFORM_ANDROID_KHR
#include <vulkan/vulkan.h>
#include "shader_blobs.hpp"
#include "scene/scene.hpp"
#include "render/forward_plus.hpp"
#include "assets/glb_reader.hpp"
#include "assets/gltf_mesh_reader.hpp"
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
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
    const float nearP = 0.1f, farP = 100.0f;
    Mat4 m{};
    m.v[0] = f / std::max(aspect, 0.01f);
    m.v[5] = -f; // Invert Y for Vulkan NDC
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

static Mat4 rotateY(float a) {
    Mat4 m = identity();
    float c = std::cos(a), s = std::sin(a);
    m.v[0] = c; m.v[2] = -s;
    m.v[8] = s; m.v[10] = c;
    return m;
}

static Mat4 rotateX(float a) {
    Mat4 m = identity();
    float c = std::cos(a), s = std::sin(a);
    m.v[5] = c; m.v[6] = s;
    m.v[9] = -s; m.v[10] = c;
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
            status_ = std::string("Vulkan initialization failed: ") + e.what();
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

    void orbit(float dx, float dy) {
        yaw_.store(yaw_.load() + dx * 0.009f);
        pitch_.store(std::clamp(pitch_.load() + dy * 0.009f, -1.35f, 1.35f));
    }

    void setAutoRotate(bool value) { autoRotate_.store(value); }

    void zoom(float delta) {
        cameraDistance_.store(std::clamp(cameraDistance_.load() + delta, 1.5f, 30.0f));
    }

    void moveCamera(float forward, float right) {
        cameraForward_.store(std::clamp(cameraForward_.load() + forward * 0.12f, -20.0f, 20.0f));
        cameraRight_.store(std::clamp(cameraRight_.load() + right * 0.12f, -20.0f, 20.0f));
    }

    void resetView() {
        yaw_.store(0.0f);
        pitch_.store(-0.28f);
        cameraDistance_.store(5.0f);
        cameraForward_.store(0.0f);
        cameraRight_.store(0.0f);
        autoRotate_.store(true);
    }

    std::string loadGlb(const std::vector<std::uint8_t>& bytes) {
        try {
            auto decoded = assets::readGlbMeshes(bytes);
            std::size_t vertexTotal = 0, indexTotal = 0;
            for (const auto& p : decoded.primitives) {
                vertexTotal += p.vertices.size();
                indexTotal += p.indices.size();
            }
            if (vertexTotal == 0 || indexTotal == 0 || vertexTotal > UINT32_MAX || indexTotal > UINT32_MAX) {
                throw std::runtime_error("GLB mesh is empty or too large for 32-bit Vulkan indices");
            }
            std::vector<Vertex> vertices; vertices.reserve(vertexTotal);
            std::vector<std::uint32_t> indices; indices.reserve(indexTotal);
            float lo[3] = {INFINITY, INFINITY, INFINITY};
            float hi[3] = {-INFINITY, -INFINITY, -INFINITY};

            for (const auto& p : decoded.primitives) {
                for (const auto& v : p.vertices) {
                    for (int k = 0; k < 3; ++k) {
                        lo[k] = std::min(lo[k], v.position[k]);
                        hi[k] = std::max(hi[k], v.position[k]);
                    }
                }
            }

            const float cx = (lo[0] + hi[0]) * 0.5f;
            const float cy = (lo[1] + hi[1]) * 0.5f;
            const float cz = (lo[2] + hi[2]) * 0.5f;
            const float span = std::max({hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]});
            if (!std::isfinite(span) || span < 1e-8f) throw std::runtime_error("GLB mesh has degenerate bounds");

            const float scale = 2.4f / span;
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
                    for (int k = 0; k < 3; ++k) {
                        out.position[k] = (v.position[k] - (k == 0 ? cx : k == 1 ? cy : cz)) * scale;
                    }
                    std::copy(v.normal, v.normal + 3, out.normal);
                    vertices.push_back(out);
                }
                for (auto index : p.indices) indices.push_back(base + index);
                ranges.push_back(range);
            }

            std::lock_guard<std::mutex> lock(gpuMutex_);
            if (!device_) throw std::runtime_error("Vulkan device is not ready; retry GLB import after viewport starts");
            check(vkDeviceWaitIdle(device_), "vkDeviceWaitIdle(GLB upload)");

            if (vertexBuffer_) vkDestroyBuffer(device_, vertexBuffer_, nullptr);
            if (indexBuffer_) vkDestroyBuffer(device_, indexBuffer_, nullptr);
            if (vertexMemory_) vkFreeMemory(device_, vertexMemory_, nullptr);
            if (indexMemory_) vkFreeMemory(device_, indexMemory_, nullptr);
            vertexBuffer_ = indexBuffer_ = VK_NULL_HANDLE;
            vertexMemory_ = indexMemory_ = VK_NULL_HANDLE;

            createBuffer(vertices.size() * sizeof(Vertex), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, vertexBuffer_, vertexMemory_, vertices.data());
            createBuffer(indices.size() * sizeof(std::uint32_t), VK_BUFFER_USAGE_INDEX_BUFFER_BIT, indexBuffer_, indexMemory_, indices.data());
            indexCount_ = static_cast<std::uint32_t>(indices.size());
            importedRanges_ = std::move(ranges);
            importedModel_ = true;

            status_ = "GLB loaded | " + std::to_string(decoded.primitives.size()) + " primitive(s) | " + std::to_string(vertexTotal) + " vertices | " + deviceName_;
            LOGI("%s", status_.c_str());
            return status_;
        } catch (const std::exception& e) {
            status_ = std::string("GLB import failed: ") + e.what();
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
    VkBuffer vertexBuffer_ = VK_NULL_HANDLE, indexBuffer_ = VK_NULL_HANDLE;
    VkDeviceMemory vertexMemory_ = VK_NULL_HANDLE, indexMemory_ = VK_NULL_HANDLE;
    uint32_t indexCount_ = 0;

    struct DrawRange {
        uint32_t firstIndex = 0, indexCount = 0;
        float baseColor[4]{1, 1, 1, 1};
        float metallic = 1.0f, roughness = 1.0f;
    };
    std::vector<DrawRange> importedRanges_;

    static constexpr size_t kFrames = 2;
    std::array<VkSemaphore, kFrames> imageAvailable_{};
    std::array<VkSemaphore, kFrames> renderFinished_{};
    std::array<VkFence, kFrames> fences_{};
    size_t frame_ = 0;

    std::atomic<bool> running_{false}, resizeRequested_{false}, autoRotate_{true};
    std::mutex gpuMutex_;
    bool importedModel_ = false;
    std::atomic<float> yaw_{0.0f}, pitch_{-0.28f};
    std::atomic<float> cameraDistance_{5.0f}, cameraForward_{0.0f}, cameraRight_{0.0f};
    std::thread thread_;
    std::chrono::steady_clock::time_point started_{};
    std::string deviceName_ = "unknown GPU", status_ = "Vulkan renderer not initialized";
    std::mutex lifecycleMutex_;

    // Exact Push Constant Structures (256-byte aligned)
    struct MeshPushConstants {
        Mat4 mvp;           // 64 bytes
        Mat4 model;         // 64 bytes
        float baseColor[4]; // 16 bytes
        float material[4];  // 16 bytes: x=metallic, y=roughness, z=ao, w=extra
        float cameraPos[4]; // 16 bytes: xyz=pos, w=time
        float sunDir[4];    // 16 bytes: xyz=dir, w=intensity
    };

    struct SkyPushConstants {
        Mat4 invViewProj;   // 64 bytes
        float sunDir[4];    // 16 bytes
    };

    void initialize() {
        VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
        app.pApplicationName = "Neo-3D";
        app.applicationVersion = VK_MAKE_VERSION(0, 3, 0);
        app.pEngineName = "Neo-3D Mobile Vulkan Engine";
        app.engineVersion = VK_MAKE_VERSION(0, 3, 0);
        app.apiVersion = VK_API_VERSION_1_0;

        const char* exts[] = {VK_KHR_SURFACE_EXTENSION_NAME, VK_KHR_ANDROID_SURFACE_EXTENSION_NAME};
        VkInstanceCreateInfo ci{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
        ci.pApplicationInfo = &app;
        ci.enabledExtensionCount = 2;
        ci.ppEnabledExtensionNames = exts;
        check(vkCreateInstance(&ci, nullptr, &instance_), "vkCreateInstance");

        VkAndroidSurfaceCreateInfoKHR si{VK_STRUCTURE_TYPE_ANDROID_SURFACE_CREATE_INFO_KHR};
        si.window = window_;
        check(vkCreateAndroidSurfaceKHR(instance_, &si, nullptr, &surface_), "vkCreateAndroidSurfaceKHR");

        selectDevice();
        createDevice();
        createGeometry();
        createSwapchain();
        chooseDepthFormat();
        createRenderPass();
        createDepthResources();
        createFramebuffers();
        createCommands();
        createGraphicsPipeline();
        createSkyPipeline();
        createSync();

        status_ = "Vulkan 3D | " + deviceName_ + " | " + std::to_string(extent_.width) + "x" + std::to_string(extent_.height) + " | Cook-Torrance GGX PBR + ACES Tone Mapping";
        LOGI("%s", status_.c_str());
    }

    void selectDevice() {
        uint32_t count = 0;
        check(vkEnumeratePhysicalDevices(instance_, &count, nullptr), "vkEnumeratePhysicalDevices(count)");
        if (!count) throw std::runtime_error("No Vulkan GPU available");
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
            if (gf == UINT32_MAX || pf == UINT32_MAX) continue;

            uint32_t ec = 0;
            vkEnumerateDeviceExtensionProperties(d, nullptr, &ec, nullptr);
            std::vector<VkExtensionProperties> es(ec);
            vkEnumerateDeviceExtensionProperties(d, nullptr, &ec, es.data());
            if (std::none_of(es.begin(), es.end(), [](const auto& e) { return std::strcmp(e.extensionName, VK_KHR_SWAPCHAIN_EXTENSION_NAME) == 0; })) continue;

            physical_ = d;
            graphicsFamily_ = gf;
            presentFamily_ = pf;
            VkPhysicalDeviceProperties p{};
            vkGetPhysicalDeviceProperties(d, &p);
            deviceName_ = p.deviceName;
            return;
        }
        throw std::runtime_error("No physical device supports graphics and Android swapchain presentation");
    }

    void createDevice() {
        float priority = 1.0f;
        std::vector<VkDeviceQueueCreateInfo> qs;
        for (uint32_t family : {graphicsFamily_, presentFamily_}) {
            if (std::any_of(qs.begin(), qs.end(), [family](const auto& q) { return q.queueFamilyIndex == family; })) continue;
            VkDeviceQueueCreateInfo q{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
            q.queueFamilyIndex = family;
            q.queueCount = 1;
            q.pQueuePriorities = &priority;
            qs.push_back(q);
        }

        VkPhysicalDeviceFeatures features{};
        VkDeviceCreateInfo ci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
        ci.queueCreateInfoCount = static_cast<uint32_t>(qs.size());
        ci.pQueueCreateInfos = qs.data();
        const char* ext = VK_KHR_SWAPCHAIN_EXTENSION_NAME;
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
        throw std::runtime_error("No Vulkan memory type matches requested properties");
    }

    void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkBuffer& buffer, VkDeviceMemory& memory, const void* data) {
        VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bi.size = size;
        bi.usage = usage;
        bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        check(vkCreateBuffer(device_, &bi, nullptr, &buffer), "vkCreateBuffer");

        VkMemoryRequirements req{};
        vkGetBufferMemoryRequirements(device_, buffer, &req);

        VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        ai.allocationSize = req.size;
        ai.memoryTypeIndex = memoryType(req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        check(vkAllocateMemory(device_, &ai, nullptr, &memory), "vkAllocateMemory(buffer)");
        check(vkBindBufferMemory(device_, buffer, memory, 0), "vkBindBufferMemory");

        if (data) {
            void* mapped = nullptr;
            check(vkMapMemory(device_, memory, 0, size, 0, &mapped), "vkMapMemory");
            std::memcpy(mapped, data, static_cast<size_t>(size));
            vkUnmapMemory(device_, memory);
        }
    }

    void createGeometry() {
        const std::array<Vertex, 24> v = {{
            {{-1, -1,  1}, { 0,  0,  1}}, {{ 1, -1,  1}, { 0,  0,  1}}, {{ 1,  1,  1}, { 0,  0,  1}}, {{-1,  1,  1}, { 0,  0,  1}},
            {{ 1, -1, -1}, { 0,  0, -1}}, {{-1, -1, -1}, { 0,  0, -1}}, {{-1,  1, -1}, { 0,  0, -1}}, {{ 1,  1, -1}, { 0,  0, -1}},
            {{-1, -1, -1}, {-1,  0,  0}}, {{-1, -1,  1}, {-1,  0,  0}}, {{-1,  1,  1}, {-1,  0,  0}}, {{-1,  1, -1}, {-1,  0,  0}},
            {{ 1, -1,  1}, { 1,  0,  0}}, {{ 1, -1, -1}, { 1,  0,  0}}, {{ 1,  1, -1}, { 1,  0,  0}}, {{ 1,  1,  1}, { 1,  0,  0}},
            {{-1,  1,  1}, { 0,  1,  0}}, {{ 1,  1,  1}, { 0,  1,  0}}, {{ 1,  1, -1}, { 0,  1,  0}}, {{-1,  1, -1}, { 0,  1,  0}},
            {{-1, -1, -1}, { 0, -1,  0}}, {{ 1, -1, -1}, { 0, -1,  0}}, {{ 1, -1,  1}, { 0, -1,  0}}, {{-1, -1,  1}, { 0, -1,  0}}
        }};
        const std::array<uint32_t, 36> idx = {{
             0,  1,  2,  2,  3,  0,
             4,  5,  6,  6,  7,  4,
             8,  9, 10, 10, 11,  8,
            12, 13, 14, 14, 15, 12,
            16, 17, 18, 18, 19, 16,
            20, 21, 22, 22, 23, 20
        }};
        indexCount_ = static_cast<uint32_t>(idx.size());
        createBuffer(sizeof(v), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, vertexBuffer_, vertexMemory_, v.data());
        createBuffer(sizeof(idx), VK_BUFFER_USAGE_INDEX_BUFFER_BIT, indexBuffer_, indexMemory_, idx.data());
    }

    void createSwapchain() {
        VkSurfaceCapabilitiesKHR caps{};
        check(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physical_, surface_, &caps), "vkGetPhysicalDeviceSurfaceCapabilitiesKHR");

        uint32_t n = 0;
        vkGetPhysicalDeviceSurfaceFormatsKHR(physical_, surface_, &n, nullptr);
        if (!n) throw std::runtime_error("No swapchain surface formats");
        std::vector<VkSurfaceFormatKHR> fs(n);
        vkGetPhysicalDeviceSurfaceFormatsKHR(physical_, surface_, &n, fs.data());

        VkSurfaceFormatKHR chosen = fs.front();
        for (const auto& f : fs) {
            if ((f.format == VK_FORMAT_R8G8B8A8_UNORM || f.format == VK_FORMAT_B8G8R8A8_UNORM) &&
                f.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
                chosen = f;
                break;
            }
        }
        format_ = chosen.format;

        if (caps.currentExtent.width != UINT32_MAX) {
            extent_ = caps.currentExtent;
        } else {
            extent_.width = std::clamp(static_cast<uint32_t>(ANativeWindow_getWidth(window_)), caps.minImageExtent.width, caps.maxImageExtent.width);
            extent_.height = std::clamp(static_cast<uint32_t>(ANativeWindow_getHeight(window_)), caps.minImageExtent.height, caps.maxImageExtent.height);
        }

        uint32_t ic = caps.minImageCount + 1;
        if (caps.maxImageCount && ic > caps.maxImageCount) ic = caps.maxImageCount;

        VkCompositeAlphaFlagBitsKHR compositeAlpha = VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR;
        if (caps.supportedCompositeAlpha & VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR) {
            compositeAlpha = VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR;
        } else if (caps.supportedCompositeAlpha & VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR) {
            compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
        }

        VkSwapchainCreateInfoKHR ci{VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR};
        ci.surface = surface_;
        ci.minImageCount = ic;
        ci.imageFormat = chosen.format;
        ci.imageColorSpace = chosen.colorSpace;
        ci.imageExtent = extent_;
        ci.imageArrayLayers = 1;
        ci.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

        uint32_t families[] = {graphicsFamily_, presentFamily_};
        if (graphicsFamily_ != presentFamily_) {
            ci.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
            ci.queueFamilyIndexCount = 2;
            ci.pQueueFamilyIndices = families;
        } else {
            ci.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
        }

        ci.preTransform = caps.currentTransform;
        ci.compositeAlpha = compositeAlpha;
        ci.presentMode = VK_PRESENT_MODE_FIFO_KHR;
        ci.clipped = VK_TRUE;

        check(vkCreateSwapchainKHR(device_, &ci, nullptr, &swapchain_), "vkCreateSwapchainKHR");

        uint32_t actual = 0;
        vkGetSwapchainImagesKHR(device_, swapchain_, &actual, nullptr);
        images_.resize(actual);
        check(vkGetSwapchainImagesKHR(device_, swapchain_, &actual, images_.data()), "vkGetSwapchainImagesKHR");

        views_.resize(images_.size());
        for (size_t i = 0; i < images_.size(); ++i) {
            VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
            vi.image = images_[i];
            vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
            vi.format = format_;
            vi.components = {VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY};
            vi.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            check(vkCreateImageView(device_, &vi, nullptr, &views_[i]), "vkCreateImageView(swapchain)");
        }
    }

    void chooseDepthFormat() {
        for (VkFormat f : {VK_FORMAT_D32_SFLOAT, VK_FORMAT_D24_UNORM_S8_UINT, VK_FORMAT_D16_UNORM}) {
            VkFormatProperties p{};
            vkGetPhysicalDeviceFormatProperties(physical_, f, &p);
            if (p.optimalTilingFeatures & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT) {
                depthFormat_ = f;
                return;
            }
        }
        throw std::runtime_error("GPU has no supported depth attachment format");
    }

    void createRenderPass() {
        std::array<VkAttachmentDescription, 2> a{};
        a[0].format = format_;
        a[0].samples = VK_SAMPLE_COUNT_1_BIT;
        a[0].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        a[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE; // Essential for Mali TBDR!
        a[0].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        a[0].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        a[0].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        a[0].finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

        a[1].format = depthFormat_;
        a[1].samples = VK_SAMPLE_COUNT_1_BIT;
        a[1].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        a[1].storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        a[1].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        a[1].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        a[1].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        a[1].finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

        VkAttachmentReference color{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
        VkAttachmentReference depth{1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};

        VkSubpassDescription sub{};
        sub.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        sub.colorAttachmentCount = 1;
        sub.pColorAttachments = &color;
        sub.pDepthStencilAttachment = &depth;

        std::array<VkSubpassDependency, 2> deps{};
        deps[0].srcSubpass = VK_SUBPASS_EXTERNAL;
        deps[0].dstSubpass = 0;
        deps[0].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
        deps[0].dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
        deps[0].srcAccessMask = 0;
        deps[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

        deps[1].srcSubpass = 0;
        deps[1].dstSubpass = VK_SUBPASS_EXTERNAL;
        deps[1].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        deps[1].dstStageMask = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
        deps[1].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        deps[1].dstAccessMask = 0;

        VkRenderPassCreateInfo ci{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
        ci.attachmentCount = static_cast<uint32_t>(a.size());
        ci.pAttachments = a.data();
        ci.subpassCount = 1;
        ci.pSubpasses = &sub;
        ci.dependencyCount = static_cast<uint32_t>(deps.size());
        ci.pDependencies = deps.data();

        check(vkCreateRenderPass(device_, &ci, nullptr, &renderPass_), "vkCreateRenderPass");
    }

    void createDepthResources() {
        depthImages_.resize(images_.size());
        depthMemory_.resize(images_.size());
        depthViews_.resize(images_.size());

        for (size_t i = 0; i < images_.size(); ++i) {
            VkImageCreateInfo ii{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
            ii.imageType = VK_IMAGE_TYPE_2D;
            ii.extent = {extent_.width, extent_.height, 1};
            ii.mipLevels = 1;
            ii.arrayLayers = 1;
            ii.format = depthFormat_;
            ii.tiling = VK_IMAGE_TILING_OPTIMAL;
            ii.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            ii.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
            ii.samples = VK_SAMPLE_COUNT_1_BIT;
            ii.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
            check(vkCreateImage(device_, &ii, nullptr, &depthImages_[i]), "vkCreateImage(depth)");

            VkMemoryRequirements req{};
            vkGetImageMemoryRequirements(device_, depthImages_[i], &req);

            VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
            ai.allocationSize = req.size;
            ai.memoryTypeIndex = memoryType(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
            check(vkAllocateMemory(device_, &ai, nullptr, &depthMemory_[i]), "vkAllocateMemory(depth)");
            check(vkBindImageMemory(device_, depthImages_[i], depthMemory_[i], 0), "vkBindImageMemory(depth)");

            VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
            vi.image = depthImages_[i];
            vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
            vi.format = depthFormat_;
            vi.subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1};
            check(vkCreateImageView(device_, &vi, nullptr, &depthViews_[i]), "vkCreateImageView(depth)");
        }
    }

    VkShaderModule shader(const uint32_t* code, size_t bytes) {
        VkShaderModuleCreateInfo ci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        ci.codeSize = bytes;
        ci.pCode = code;
        VkShaderModule m = VK_NULL_HANDLE;
        check(vkCreateShaderModule(device_, &ci, nullptr, &m), "vkCreateShaderModule");
        return m;
    }

    void createGraphicsPipeline() {
        VkShaderModule vert = shader(kMeshVert, kMeshVertSize);
        VkShaderModule frag = shader(kMeshFrag, kMeshFragSize);
        try {
            VkPushConstantRange range{};
            range.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
            range.offset = 0;
            range.size = sizeof(MeshPushConstants);

            VkPipelineLayoutCreateInfo li{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
            li.pushConstantRangeCount = 1;
            li.pPushConstantRanges = &range;
            check(vkCreatePipelineLayout(device_, &li, nullptr, &pipelineLayout_), "vkCreatePipelineLayout");

            std::array<VkPipelineShaderStageCreateInfo, 2> stages{};
            stages[0] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
            stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
            stages[0].module = vert;
            stages[0].pName = "main";
            stages[1] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
            stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
            stages[1].module = frag;
            stages[1].pName = "main";

            VkVertexInputBindingDescription binding{0, sizeof(Vertex), VK_VERTEX_INPUT_RATE_VERTEX};
            std::array<VkVertexInputAttributeDescription, 2> attrs = {
                VkVertexInputAttributeDescription{0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, position)},
                VkVertexInputAttributeDescription{1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, normal)}
            };

            VkPipelineVertexInputStateCreateInfo vi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
            vi.vertexBindingDescriptionCount = 1;
            vi.pVertexBindingDescriptions = &binding;
            vi.vertexAttributeDescriptionCount = static_cast<uint32_t>(attrs.size());
            vi.pVertexAttributeDescriptions = attrs.data();

            VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
            ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

            VkViewport vp{0.0f, 0.0f, static_cast<float>(extent_.width), static_cast<float>(extent_.height), 0.0f, 1.0f};
            VkRect2D sc{{0, 0}, extent_};
            VkPipelineViewportStateCreateInfo vsi{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
            vsi.viewportCount = 1;
            vsi.pViewports = &vp;
            vsi.scissorCount = 1;
            vsi.pScissors = &sc;

            VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
            rs.polygonMode = VK_POLYGON_MODE_FILL;
            rs.cullMode = VK_CULL_MODE_NONE;
            rs.frontFace = VK_FRONT_FACE_CLOCKWISE;
            rs.lineWidth = 1.0f;

            VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
            ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

            VkPipelineDepthStencilStateCreateInfo ds{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
            ds.depthTestEnable = VK_TRUE;
            ds.depthWriteEnable = VK_TRUE;
            ds.depthCompareOp = VK_COMPARE_OP_LESS;
            ds.minDepthBounds = 0.0f;
            ds.maxDepthBounds = 1.0f;

            VkPipelineColorBlendAttachmentState ba{};
            ba.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

            VkPipelineColorBlendStateCreateInfo bs{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
            bs.attachmentCount = 1;
            bs.pAttachments = &ba;

            VkGraphicsPipelineCreateInfo pi{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
            pi.stageCount = 2;
            pi.pStages = stages.data();
            pi.pVertexInputState = &vi;
            pi.pInputAssemblyState = &ia;
            pi.pViewportState = &vsi;
            pi.pRasterizationState = &rs;
            pi.pMultisampleState = &ms;
            pi.pDepthStencilState = &ds;
            pi.pColorBlendState = &bs;
            pi.layout = pipelineLayout_;
            pi.renderPass = renderPass_;
            pi.subpass = 0;

            check(vkCreateGraphicsPipelines(device_, VK_NULL_HANDLE, 1, &pi, nullptr, &pipeline_), "vkCreateGraphicsPipelines");
        } catch (...) {
            vkDestroyShaderModule(device_, vert, nullptr);
            vkDestroyShaderModule(device_, frag, nullptr);
            throw;
        }
        vkDestroyShaderModule(device_, vert, nullptr);
        vkDestroyShaderModule(device_, frag, nullptr);
    }

    void createFramebuffers() {
        framebuffers_.resize(views_.size());
        for (size_t i = 0; i < views_.size(); ++i) {
            VkImageView atts[] = {views_[i], depthViews_[i]};
            VkFramebufferCreateInfo ci{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
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
        VkCommandPoolCreateInfo pi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        pi.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        pi.queueFamilyIndex = graphicsFamily_;
        check(vkCreateCommandPool(device_, &pi, nullptr, &commandPool_), "vkCreateCommandPool");

        commandBuffers_.resize(framebuffers_.size());
        VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        ai.commandPool = commandPool_;
        ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        ai.commandBufferCount = static_cast<uint32_t>(commandBuffers_.size());
        check(vkAllocateCommandBuffers(device_, &ai, commandBuffers_.data()), "vkAllocateCommandBuffers");
    }

    void createSkyPipeline() {
        VkShaderModule vert = shader(kSkyVert, kSkyVertSize);
        VkShaderModule frag = shader(kSkyFrag, kSkyFragSize);
        try {
            VkPushConstantRange range{};
            range.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
            range.offset = 0;
            range.size = sizeof(SkyPushConstants);

            VkPipelineLayoutCreateInfo li{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
            li.pushConstantRangeCount = 1;
            li.pPushConstantRanges = &range;
            check(vkCreatePipelineLayout(device_, &li, nullptr, &skyPipelineLayout_), "vkCreatePipelineLayout(sky)");

            std::array<VkPipelineShaderStageCreateInfo, 2> stages{};
            stages[0] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
            stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
            stages[0].module = vert;
            stages[0].pName = "main";
            stages[1] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
            stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
            stages[1].module = frag;
            stages[1].pName = "main";

            VkPipelineVertexInputStateCreateInfo vi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
            VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
            ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

            VkViewport vp{0.0f, 0.0f, static_cast<float>(extent_.width), static_cast<float>(extent_.height), 0.0f, 1.0f};
            VkRect2D sc{{0, 0}, extent_};
            VkPipelineViewportStateCreateInfo vsi{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
            vsi.viewportCount = 1;
            vsi.pViewports = &vp;
            vsi.scissorCount = 1;
            vsi.pScissors = &sc;

            VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
            rs.polygonMode = VK_POLYGON_MODE_FILL;
            rs.cullMode = VK_CULL_MODE_NONE;
            rs.lineWidth = 1.0f;

            VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
            ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

            VkPipelineDepthStencilStateCreateInfo ds{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
            ds.depthTestEnable = VK_FALSE;
            ds.depthWriteEnable = VK_FALSE;

            VkPipelineColorBlendAttachmentState ba{};
            ba.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

            VkPipelineColorBlendStateCreateInfo bs{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
            bs.attachmentCount = 1;
            bs.pAttachments = &ba;

            VkGraphicsPipelineCreateInfo pi{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
            pi.stageCount = 2;
            pi.pStages = stages.data();
            pi.pVertexInputState = &vi;
            pi.pInputAssemblyState = &ia;
            pi.pViewportState = &vsi;
            pi.pRasterizationState = &rs;
            pi.pMultisampleState = &ms;
            pi.pDepthStencilState = &ds;
            pi.pColorBlendState = &bs;
            pi.layout = skyPipelineLayout_;
            pi.renderPass = renderPass_;

            check(vkCreateGraphicsPipelines(device_, VK_NULL_HANDLE, 1, &pi, nullptr, &skyPipeline_), "vkCreateGraphicsPipelines(sky)");
        } catch (...) {
            if (vert) vkDestroyShaderModule(device_, vert, nullptr);
            if (frag) vkDestroyShaderModule(device_, frag, nullptr);
            throw;
        }
        vkDestroyShaderModule(device_, vert, nullptr);
        vkDestroyShaderModule(device_, frag, nullptr);
    }

    void createSync() {
        VkSemaphoreCreateInfo si{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
        VkFenceCreateInfo fi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        fi.flags = VK_FENCE_CREATE_SIGNALED_BIT;
        for (size_t i = 0; i < kFrames; ++i) {
            check(vkCreateSemaphore(device_, &si, nullptr, &imageAvailable_[i]), "vkCreateSemaphore");
            check(vkCreateSemaphore(device_, &si, nullptr, &renderFinished_[i]), "vkCreateSemaphore");
            check(vkCreateFence(device_, &fi, nullptr, &fences_[i]), "vkCreateFence");
        }
    }

    void record(uint32_t i) {
        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        check(vkBeginCommandBuffer(commandBuffers_[i], &bi), "vkBeginCommandBuffer");

        VkClearValue clears[2]{};
        clears[0].color = {{0.12f, 0.22f, 0.38f, 1.0f}};
        clears[1].depthStencil = {1.0f, 0};

        VkRenderPassBeginInfo rp{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
        rp.renderPass = renderPass_;
        rp.framebuffer = framebuffers_[i];
        rp.renderArea = {{0, 0}, extent_};
        rp.clearValueCount = 2;
        rp.pClearValues = clears;

        vkCmdBeginRenderPass(commandBuffers_[i], &rp, VK_SUBPASS_CONTENTS_INLINE);

        float t = std::chrono::duration<float>(std::chrono::steady_clock::now() - started_).count();
        float yaw = autoRotate_.load() ? t * 0.45f : yaw_.load();
        float pitch = pitch_.load();
        float dist = cameraDistance_.load();

        // 1. Precise 3D Camera Orbit Position (World Space)
        float camX = dist * std::sin(yaw) * std::cos(pitch) + cameraRight_.load();
        float camY = -dist * std::sin(pitch) + cameraForward_.load();
        float camZ = dist * std::cos(yaw) * std::cos(pitch);

        Mat4 view = multiply(rotateX(-pitch), multiply(rotateY(-yaw), translate(-cameraRight_.load(), -cameraForward_.load(), -dist)));
        Mat4 projection = perspective(static_cast<float>(extent_.width) / static_cast<float>(std::max(1u, extent_.height)));
        Mat4 viewProj = multiply(projection, view);

        // Sun Direction & Intensity
        float sunDir[4] = {-0.55f, 0.85f, 0.45f, 3.4f};

        // 2. Draw 3D Atmospheric Sky Dome (Rotates accurately with camera!)
        if (skyPipeline_ != VK_NULL_HANDLE) {
            SkyPushConstants skyPush{};
            skyPush.invViewProj = inverseMat4(viewProj);
            std::copy(sunDir, sunDir + 4, skyPush.sunDir);

            vkCmdBindPipeline(commandBuffers_[i], VK_PIPELINE_BIND_POINT_GRAPHICS, skyPipeline_);
            vkCmdPushConstants(commandBuffers_[i], skyPipelineLayout_, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(SkyPushConstants), &skyPush);
            vkCmdDraw(commandBuffers_[i], 3, 1, 0, 0);
        }

        // 3. Draw 3D Geometry with Cook-Torrance GGX PBR Shading
        vkCmdBindPipeline(commandBuffers_[i], VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_);
        VkDeviceSize offset = 0;
        vkCmdBindVertexBuffers(commandBuffers_[i], 0, 1, &vertexBuffer_, &offset);
        vkCmdBindIndexBuffer(commandBuffers_[i], indexBuffer_, 0, VK_INDEX_TYPE_UINT32);

        MeshPushConstants push{};
        push.cameraPos[0] = camX;
        push.cameraPos[1] = camY;
        push.cameraPos[2] = camZ;
        push.cameraPos[3] = t;
        std::copy(sunDir, sunDir + 4, push.sunDir);

        if (importedModel_) {
            Mat4 model = identity();
            push.model = model;
            push.mvp = multiply(viewProj, model);

            for (const auto& range : importedRanges_) {
                std::copy(range.baseColor, range.baseColor + 4, push.baseColor);
                push.material[0] = range.metallic;
                push.material[1] = range.roughness;
                push.material[2] = 1.0f; // AO
                push.material[3] = 0.0f;

                vkCmdPushConstants(commandBuffers_[i], pipelineLayout_, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(MeshPushConstants), &push);
                vkCmdDrawIndexed(commandBuffers_[i], range.indexCount, 1, range.firstIndex, 0, 0);
            }
        } else {
            const float positions[3] = {-1.55f, 0.0f, 1.55f};
            for (int object = 0; object < 3; ++object) {
                Mat4 local = multiply(rotateY(yaw + (object - 1) * 0.45f), rotateX(pitch + (object - 1) * 0.12f));
                Mat4 model = multiply(translate(positions[object], object == 1 ? 0.0f : -0.12f, 0.0f), local);
                push.model = model;
                push.mvp = multiply(viewProj, model);

                // Material properties: Gold, Chrome, and Rough Matte
                if (object == 0) {
                    push.baseColor[0] = 1.00f; push.baseColor[1] = 0.76f; push.baseColor[2] = 0.33f; push.baseColor[3] = 1.0f; // Gold
                    push.material[0] = 0.95f; push.material[1] = 0.15f; push.material[2] = 1.0f; push.material[3] = 0.0f;
                } else if (object == 1) {
                    push.baseColor[0] = 0.95f; push.baseColor[1] = 0.95f; push.baseColor[2] = 0.98f; push.baseColor[3] = 1.0f; // Silver
                    push.material[0] = 0.98f; push.material[1] = 0.08f; push.material[2] = 1.0f; push.material[3] = 0.0f;
                } else {
                    push.baseColor[0] = 0.18f; push.baseColor[1] = 0.52f; push.baseColor[2] = 0.88f; push.baseColor[3] = 1.0f; // Matte Blue
                    push.material[0] = 0.05f; push.material[1] = 0.45f; push.material[2] = 1.0f; push.material[3] = 0.0f;
                }

                vkCmdPushConstants(commandBuffers_[i], pipelineLayout_, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(MeshPushConstants), &push);
                vkCmdDrawIndexed(commandBuffers_[i], indexCount_, 1, 0, 0, 0);
            }
        }

        vkCmdEndRenderPass(commandBuffers_[i]);
        check(vkEndCommandBuffer(commandBuffers_[i]), "vkEndCommandBuffer");
    }

    void renderLoop() {
        try {
            LOGI("Vulkan render loop started");
            while (running_.load()) {
                if (resizeRequested_.exchange(false)) {
                    std::lock_guard<std::mutex> lock(lifecycleMutex_);
                    rebuild();
                }

                if (!swapchain_ || !device_ || !pipeline_) {
                    status_ = "Vulkan renderer stopped: swapchain, device, or graphics pipeline is unavailable";
                    LOGE("%s", status_.c_str());
                    break;
                }

                std::lock_guard<std::mutex> gpuLock(gpuMutex_);
                check(vkWaitForFences(device_, 1, &fences_[frame_], VK_TRUE, UINT64_MAX), "vkWaitForFences");

                uint32_t imageIndex = 0;
                VkResult ac = vkAcquireNextImageKHR(device_, swapchain_, UINT64_MAX, imageAvailable_[frame_], VK_NULL_HANDLE, &imageIndex);

                if (ac == VK_ERROR_OUT_OF_DATE_KHR) {
                    resizeRequested_.store(true);
                    continue;
                }
                if (ac != VK_SUCCESS && ac != VK_SUBOPTIMAL_KHR) {
                    status_ = "Vulkan frame error: vkAcquireNextImageKHR result=" + std::to_string(static_cast<int>(ac));
                    LOGE("%s", status_.c_str());
                    break;
                }

                vkResetFences(device_, 1, &fences_[frame_]);
                vkResetCommandBuffer(commandBuffers_[imageIndex], 0);
                record(imageIndex);

                VkPipelineStageFlags wait = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
                VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
                si.waitSemaphoreCount = 1;
                si.pWaitSemaphores = &imageAvailable_[frame_];
                si.pWaitDstStageMask = &wait;
                si.commandBufferCount = 1;
                si.pCommandBuffers = &commandBuffers_[imageIndex];
                si.signalSemaphoreCount = 1;
                si.pSignalSemaphores = &renderFinished_[frame_];

                VkResult sub = vkQueueSubmit(graphicsQueue_, 1, &si, fences_[frame_]);
                if (sub != VK_SUCCESS) {
                    status_ = "Vulkan frame error: vkQueueSubmit result=" + std::to_string(static_cast<int>(sub));
                    LOGE("%s", status_.c_str());
                    break;
                }

                VkPresentInfoKHR pi{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
                pi.waitSemaphoreCount = 1;
                pi.pWaitSemaphores = &renderFinished_[frame_];
                pi.swapchainCount = 1;
                pi.pSwapchains = &swapchain_;
                pi.pImageIndices = &imageIndex;

                VkResult pr = vkQueuePresentKHR(presentQueue_, &pi);
                if (pr == VK_ERROR_OUT_OF_DATE_KHR || pr == VK_SUBOPTIMAL_KHR || ac == VK_SUBOPTIMAL_KHR) {
                    resizeRequested_.store(true);
                } else if (pr != VK_SUCCESS) {
                    status_ = "Vulkan presentation failed: VkResult=" + std::to_string(static_cast<int>(pr));
                    LOGE("%s", status_.c_str());
                    break;
                }

                frame_ = (frame_ + 1) % kFrames;
            }
        } catch (const std::exception& e) {
            status_ = std::string("Vulkan render-loop failure: ") + e.what();
            LOGE("%s", status_.c_str());
            running_.store(false);
        } catch (...) {
            status_ = "Vulkan render-loop failure: unknown native exception";
            LOGE("%s", status_.c_str());
            running_.store(false);
        }
    }

    void rebuild() {
        if (!device_) return;
        std::lock_guard<std::mutex> gpuLock(gpuMutex_);
        vkDeviceWaitIdle(device_);
        destroySwapchain();
        try {
            createSwapchain();
            chooseDepthFormat();
            createRenderPass();
            createDepthResources();
            createFramebuffers();
            createCommands();
            createGraphicsPipeline();
            createSkyPipeline();
            status_ = "Vulkan 3D | " + deviceName_ + " | " + std::to_string(extent_.width) + "x" + std::to_string(extent_.height) + " | Cook-Torrance GGX PBR + ACES Tone Mapping";
        } catch (const std::exception& e) {
            status_ = std::string("Swapchain rebuild failed: ") + e.what();
            LOGE("%s", status_.c_str());
        }
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
            if (vertexBuffer_) vkDestroyBuffer(device_, vertexBuffer_, nullptr);
            if (indexBuffer_) vkDestroyBuffer(device_, indexBuffer_, nullptr);
            if (vertexMemory_) vkFreeMemory(device_, vertexMemory_, nullptr);
            if (indexMemory_) vkFreeMemory(device_, indexMemory_, nullptr);

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

extern "C" JNIEXPORT void JNICALL Java_com_neo3d_engine_MainActivity_nativeStart(JNIEnv* env, jobject, jobject surface) {
    std::lock_guard<std::mutex> lock(neo3d::gMutex);
    if (neo3d::gRenderer) {
        neo3d::gRenderer->stop();
        delete neo3d::gRenderer;
        neo3d::gRenderer = nullptr;
    }
    ANativeWindow* w = ANativeWindow_fromSurface(env, surface);
    if (!w) {
        LOGE("ANativeWindow_fromSurface returned null");
        return;
    }
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
    std::string s = neo3d::gRenderer ? neo3d::gRenderer->status() : "Vulkan renderer stopped";
    return env->NewStringUTF(s.c_str());
}

extern "C" JNIEXPORT jstring JNICALL Java_com_neo3d_engine_MainActivity_nativeLoadGlb(JNIEnv* env, jobject, jbyteArray data) {
    if (!data) return env->NewStringUTF("GLB import failed: no file data");
    const jsize length = env->GetArrayLength(data);
    if (length <= 0 || length > 128 * 1024 * 1024) return env->NewStringUTF("GLB import failed: file is empty or exceeds 128 MiB");
    std::vector<std::uint8_t> bytes(static_cast<std::size_t>(length));
    env->GetByteArrayRegion(data, 0, length, reinterpret_cast<jbyte*>(bytes.data()));
    if (env->ExceptionCheck()) return env->NewStringUTF("GLB import failed: could not read selected file");

    std::lock_guard<std::mutex> lock(neo3d::gMutex);
    if (!neo3d::gRenderer) return env->NewStringUTF("GLB import failed: start the viewport first");
    const std::string result = neo3d::gRenderer->loadGlb(bytes);
    return env->NewStringUTF(result.c_str());
}

extern "C" JNIEXPORT void JNICALL Java_com_neo3d_engine_MainActivity_nativeOrbit(JNIEnv*, jobject, jfloat dx, jfloat dy) {
    std::lock_guard<std::mutex> lock(neo3d::gMutex);
    if (neo3d::gRenderer) neo3d::gRenderer->orbit(dx, dy);
}

extern "C" JNIEXPORT void JNICALL Java_com_neo3d_engine_MainActivity_nativeZoom(JNIEnv*, jobject, jfloat delta) {
    std::lock_guard<std::mutex> lock(neo3d::gMutex);
    if (neo3d::gRenderer) neo3d::gRenderer->zoom(delta);
}

extern "C" JNIEXPORT void JNICALL Java_com_neo3d_engine_MainActivity_nativeMoveCamera(JNIEnv*, jobject, jfloat forward, jfloat right) {
    std::lock_guard<std::mutex> lock(neo3d::gMutex);
    if (neo3d::gRenderer) neo3d::gRenderer->moveCamera(forward, right);
}

extern "C" JNIEXPORT void JNICALL Java_com_neo3d_engine_MainActivity_nativeSetAutoRotate(JNIEnv*, jobject, jboolean enabled) {
    std::lock_guard<std::mutex> lock(neo3d::gMutex);
    if (neo3d::gRenderer) neo3d::gRenderer->setAutoRotate(enabled == JNI_TRUE);
}

extern "C" JNIEXPORT void JNICALL Java_com_neo3d_engine_MainActivity_nativeResetView(JNIEnv*, jobject) {
    std::lock_guard<std::mutex> lock(neo3d::gMutex);
    if (neo3d::gRenderer) neo3d::gRenderer->resetView();
}
