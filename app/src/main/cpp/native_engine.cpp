#include <jni.h>
#include <android/native_window.h>
#include <android/native_window_jni.h>
#include <android/log.h>
#include <vulkan/vulkan.h>
#include "scene/scene.hpp"
#include "render/forward_plus.hpp"
#include "assets/glb_reader.hpp"
#include <algorithm>
#include <array>
#include <atomic>
#include <cstring>
#include <mutex>
#include <string>
#include <stdexcept>
#include <utility>
#include <thread>
#include <vector>

#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, "Neo3D", __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, "Neo3D", __VA_ARGS__)

namespace neo3d {
static const char* kSwapchainExtension = VK_KHR_SWAPCHAIN_EXTENSION_NAME;
static void check(VkResult result, const char* operation) {
    if (result != VK_SUCCESS) { LOGE("%s failed: VkResult=%d", operation, static_cast<int>(result)); throw std::runtime_error(operation); }
}

class VulkanRenderer {
public:
    explicit VulkanRenderer(ANativeWindow* window) : window_(window) { if (window_) ANativeWindow_acquire(window_); }
    ~VulkanRenderer() { stop(); if (window_) ANativeWindow_release(window_); }
    void start() {
        try { initialize(); running_.store(true); thread_ = std::thread(&VulkanRenderer::renderLoop, this); }
        catch (const std::exception& e) { status_ = std::string("Vulkan initialization failed: ") + e.what(); LOGE("%s", status_.c_str()); cleanup(); }
    }
    void resize(int width, int height) { (void)width; (void)height; resizeRequested_.store(true); }
    void stop() {
        running_.store(false);
        if (thread_.joinable()) thread_.join();
        cleanup();
    }
    std::string status() const { return status_; }
private:
    ANativeWindow* window_ = nullptr;
    VkInstance instance_ = VK_NULL_HANDLE;
    VkSurfaceKHR surface_ = VK_NULL_HANDLE;
    VkPhysicalDevice physical_ = VK_NULL_HANDLE;
    VkDevice device_ = VK_NULL_HANDLE;
    uint32_t graphicsFamily_ = UINT32_MAX, presentFamily_ = UINT32_MAX;
    VkQueue graphicsQueue_ = VK_NULL_HANDLE, presentQueue_ = VK_NULL_HANDLE;
    VkSwapchainKHR swapchain_ = VK_NULL_HANDLE;
    VkFormat format_ = VK_FORMAT_UNDEFINED;
    VkExtent2D extent_{};
    std::vector<VkImage> images_;
    std::vector<VkImageView> views_;
    VkRenderPass renderPass_ = VK_NULL_HANDLE;
    std::vector<VkFramebuffer> framebuffers_;
    VkCommandPool commandPool_ = VK_NULL_HANDLE;
    std::vector<VkCommandBuffer> commandBuffers_;
    static constexpr size_t kFrames = 2;
    std::array<VkSemaphore, kFrames> imageAvailable_{};
    std::array<VkSemaphore, kFrames> renderFinished_{};
    std::array<VkFence, kFrames> fences_{};
    size_t frame_ = 0;
    std::atomic<bool> running_{false}, resizeRequested_{false};
    std::thread thread_;
    std::string status_ = "Vulkan renderer not initialized";
    std::mutex lifecycleMutex_;

    void initialize() {
        VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
        app.pApplicationName = "Neo-3D"; app.applicationVersion = VK_MAKE_VERSION(0,1,0);
        app.pEngineName = "Neo-3D Native Engine"; app.engineVersion = VK_MAKE_VERSION(0,1,0); app.apiVersion = VK_API_VERSION_1_0;
        const char* extensions[] = {VK_KHR_SURFACE_EXTENSION_NAME, VK_KHR_ANDROID_SURFACE_EXTENSION_NAME};
        VkInstanceCreateInfo ci{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO}; ci.pApplicationInfo = &app; ci.enabledExtensionCount = 2; ci.ppEnabledExtensionNames = extensions;
        check(vkCreateInstance(&ci, nullptr, &instance_), "vkCreateInstance");
        VkAndroidSurfaceCreateInfoKHR surfaceInfo{VK_STRUCTURE_TYPE_ANDROID_SURFACE_CREATE_INFO_KHR}; surfaceInfo.window = window_;
        check(vkCreateAndroidSurfaceKHR(instance_, &surfaceInfo, nullptr, &surface_), "vkCreateAndroidSurfaceKHR");
        selectDevice(); createDevice(); createSwapchain(); createRenderPass(); createFramebuffers(); createCommands(); createSync();
        status_ = "Vulkan active | " + deviceName_ + " | swapchain " + std::to_string(extent_.width) + "x" + std::to_string(extent_.height);
        LOGI("%s", status_.c_str());
    }
    std::string deviceName_ = "unknown GPU";
    void selectDevice() {
        uint32_t count = 0; check(vkEnumeratePhysicalDevices(instance_, &count, nullptr), "vkEnumeratePhysicalDevices(count)");
        if (!count) throw std::runtime_error("No Vulkan-capable GPU detected on this device");
        std::vector<VkPhysicalDevice> devices(count); check(vkEnumeratePhysicalDevices(instance_, &count, devices.data()), "vkEnumeratePhysicalDevices");
        for (auto candidate : devices) {
            uint32_t qcount = 0; vkGetPhysicalDeviceQueueFamilyProperties(candidate, &qcount, nullptr);
            std::vector<VkQueueFamilyProperties> queues(qcount); vkGetPhysicalDeviceQueueFamilyProperties(candidate, &qcount, queues.data());
            uint32_t graphics = UINT32_MAX, present = UINT32_MAX;
            for (uint32_t i = 0; i < qcount; ++i) {
                if (queues[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) graphics = i;
                VkBool32 supported = VK_FALSE; vkGetPhysicalDeviceSurfaceSupportKHR(candidate, i, surface_, &supported);
                if (supported) present = i;
            }
            if (graphics == UINT32_MAX || present == UINT32_MAX) continue;
            uint32_t extCount = 0; vkEnumerateDeviceExtensionProperties(candidate, nullptr, &extCount, nullptr);
            std::vector<VkExtensionProperties> exts(extCount); vkEnumerateDeviceExtensionProperties(candidate, nullptr, &extCount, exts.data());
            bool swapchainSupported = std::any_of(exts.begin(), exts.end(), [](const auto& e){ return std::strcmp(e.extensionName, kSwapchainExtension) == 0; });
            if (!swapchainSupported) continue;
            physical_ = candidate; graphicsFamily_ = graphics; presentFamily_ = present;
            VkPhysicalDeviceProperties props{}; vkGetPhysicalDeviceProperties(candidate, &props); deviceName_ = props.deviceName; return;
        }
        throw std::runtime_error("No GPU supports graphics, Android surface presentation and VK_KHR_swapchain");
    }
    void createDevice() {
        float priority = 1.0f; std::vector<VkDeviceQueueCreateInfo> queues;
        for (uint32_t family : {graphicsFamily_, presentFamily_}) {
            if (std::any_of(queues.begin(), queues.end(), [family](const auto& q){ return q.queueFamilyIndex == family; })) continue;
            VkDeviceQueueCreateInfo q{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO}; q.queueFamilyIndex = family; q.queueCount = 1; q.pQueuePriorities = &priority; queues.push_back(q);
        }
        VkPhysicalDeviceFeatures features{}; VkDeviceCreateInfo ci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
        ci.queueCreateInfoCount = static_cast<uint32_t>(queues.size()); ci.pQueueCreateInfos = queues.data(); ci.enabledExtensionCount = 1; ci.ppEnabledExtensionNames = &kSwapchainExtension; ci.pEnabledFeatures = &features;
        check(vkCreateDevice(physical_, &ci, nullptr, &device_), "vkCreateDevice");
        vkGetDeviceQueue(device_, graphicsFamily_, 0, &graphicsQueue_); vkGetDeviceQueue(device_, presentFamily_, 0, &presentQueue_);
    }
    void createSwapchain() {
        VkSurfaceCapabilitiesKHR caps{}; check(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physical_, surface_, &caps), "vkGetPhysicalDeviceSurfaceCapabilitiesKHR");
        uint32_t formatCount = 0; vkGetPhysicalDeviceSurfaceFormatsKHR(physical_, surface_, &formatCount, nullptr);
        if (!formatCount) throw std::runtime_error("Surface exposes no Vulkan formats");
        std::vector<VkSurfaceFormatKHR> formats(formatCount); vkGetPhysicalDeviceSurfaceFormatsKHR(physical_, surface_, &formatCount, formats.data());
        VkSurfaceFormatKHR chosen = formats.front();
        for (const auto& f : formats) if (f.format == VK_FORMAT_B8G8R8A8_UNORM && f.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) { chosen = f; break; }
        format_ = chosen.format;
        if (caps.currentExtent.width != UINT32_MAX) extent_ = caps.currentExtent;
        else { extent_.width = std::clamp(static_cast<uint32_t>(ANativeWindow_getWidth(window_)), caps.minImageExtent.width, caps.maxImageExtent.width); extent_.height = std::clamp(static_cast<uint32_t>(ANativeWindow_getHeight(window_)), caps.minImageExtent.height, caps.maxImageExtent.height); }
        uint32_t imageCount = caps.minImageCount + 1; if (caps.maxImageCount && imageCount > caps.maxImageCount) imageCount = caps.maxImageCount;
        VkPresentModeKHR mode = VK_PRESENT_MODE_FIFO_KHR;
        VkSwapchainCreateInfoKHR ci{VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR}; ci.surface = surface_; ci.minImageCount = imageCount; ci.imageFormat = chosen.format; ci.imageColorSpace = chosen.colorSpace; ci.imageExtent = extent_; ci.imageArrayLayers = 1; ci.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
        uint32_t familyIndices[] = {graphicsFamily_, presentFamily_};
        if (graphicsFamily_ != presentFamily_) { ci.imageSharingMode = VK_SHARING_MODE_CONCURRENT; ci.queueFamilyIndexCount = 2; ci.pQueueFamilyIndices = familyIndices; } else ci.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
        ci.preTransform = caps.currentTransform; ci.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR; ci.presentMode = mode; ci.clipped = VK_TRUE;
        check(vkCreateSwapchainKHR(device_, &ci, nullptr, &swapchain_), "vkCreateSwapchainKHR");
        uint32_t actual = 0; vkGetSwapchainImagesKHR(device_, swapchain_, &actual, nullptr); images_.resize(actual); vkGetSwapchainImagesKHR(device_, swapchain_, &actual, images_.data());
        views_.resize(images_.size());
        for (size_t i = 0; i < images_.size(); ++i) { VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO}; vi.image = images_[i]; vi.viewType = VK_IMAGE_VIEW_TYPE_2D; vi.format = format_; vi.components = {VK_COMPONENT_SWIZZLE_IDENTITY,VK_COMPONENT_SWIZZLE_IDENTITY,VK_COMPONENT_SWIZZLE_IDENTITY,VK_COMPONENT_SWIZZLE_IDENTITY}; vi.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT,0,1,0,1}; check(vkCreateImageView(device_, &vi, nullptr, &views_[i]), "vkCreateImageView"); }
    }
    void createRenderPass() {
        VkAttachmentDescription color{}; color.format = format_; color.samples = VK_SAMPLE_COUNT_1_BIT; color.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR; color.storeOp = VK_ATTACHMENT_STORE_OP_STORE; color.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE; color.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE; color.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED; color.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
        VkAttachmentReference ref{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL}; VkSubpassDescription sub{}; sub.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS; sub.colorAttachmentCount = 1; sub.pColorAttachments = &ref;
        VkSubpassDependency dep{}; dep.srcSubpass = VK_SUBPASS_EXTERNAL; dep.dstSubpass = 0; dep.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT; dep.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT; dep.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        VkRenderPassCreateInfo ci{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO}; ci.attachmentCount = 1; ci.pAttachments = &color; ci.subpassCount = 1; ci.pSubpasses = &sub; ci.dependencyCount = 1; ci.pDependencies = &dep; check(vkCreateRenderPass(device_, &ci, nullptr, &renderPass_), "vkCreateRenderPass");
    }
    void createFramebuffers() {
        framebuffers_.resize(views_.size()); for (size_t i=0; i<views_.size(); ++i) { VkFramebufferCreateInfo ci{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO}; ci.renderPass=renderPass_; ci.attachmentCount=1; ci.pAttachments=&views_[i]; ci.width=extent_.width; ci.height=extent_.height; ci.layers=1; check(vkCreateFramebuffer(device_, &ci, nullptr, &framebuffers_[i]), "vkCreateFramebuffer"); }
    }
    void createCommands() {
        VkCommandPoolCreateInfo pi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO}; pi.flags=VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT; pi.queueFamilyIndex=graphicsFamily_; check(vkCreateCommandPool(device_, &pi, nullptr, &commandPool_), "vkCreateCommandPool");
        commandBuffers_.resize(framebuffers_.size()); VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO}; ai.commandPool=commandPool_; ai.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY; ai.commandBufferCount=static_cast<uint32_t>(commandBuffers_.size()); check(vkAllocateCommandBuffers(device_, &ai, commandBuffers_.data()), "vkAllocateCommandBuffers");
    }
    void createSync() {
        VkSemaphoreCreateInfo si{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO}; VkFenceCreateInfo fi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO}; fi.flags=VK_FENCE_CREATE_SIGNALED_BIT;
        for (size_t i=0; i<kFrames; ++i) { check(vkCreateSemaphore(device_, &si, nullptr, &imageAvailable_[i]), "vkCreateSemaphore(imageAvailable)"); check(vkCreateSemaphore(device_, &si, nullptr, &renderFinished_[i]), "vkCreateSemaphore(renderFinished)"); check(vkCreateFence(device_, &fi, nullptr, &fences_[i]), "vkCreateFence"); }
    }
    void record(uint32_t index) {
        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO}; check(vkBeginCommandBuffer(commandBuffers_[index], &bi), "vkBeginCommandBuffer");
        VkClearValue clear{}; clear.color = {{0.025f, 0.045f, 0.075f, 1.0f}};
        VkRenderPassBeginInfo rp{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO}; rp.renderPass=renderPass_; rp.framebuffer=framebuffers_[index]; rp.renderArea={{0,0},extent_}; rp.clearValueCount=1; rp.pClearValues=&clear;
        vkCmdBeginRenderPass(commandBuffers_[index], &rp, VK_SUBPASS_CONTENTS_INLINE); vkCmdEndRenderPass(commandBuffers_[index]); check(vkEndCommandBuffer(commandBuffers_[index]), "vkEndCommandBuffer");
    }
    void renderLoop() {
        while (running_.load()) {
            if (resizeRequested_.exchange(false)) { std::lock_guard<std::mutex> lock(lifecycleMutex_); rebuild(); }
            if (swapchain_ == VK_NULL_HANDLE || device_ == VK_NULL_HANDLE) break;
            vkWaitForFences(device_, 1, &fences_[frame_], VK_TRUE, UINT64_MAX);
            uint32_t imageIndex = 0; VkResult acquire = vkAcquireNextImageKHR(device_, swapchain_, UINT64_MAX, imageAvailable_[frame_], VK_NULL_HANDLE, &imageIndex);
            if (acquire == VK_ERROR_OUT_OF_DATE_KHR) { std::lock_guard<std::mutex> lock(lifecycleMutex_); rebuild(); continue; }
            if (acquire != VK_SUCCESS && acquire != VK_SUBOPTIMAL_KHR) { LOGE("vkAcquireNextImageKHR: %d", acquire); break; }
            vkResetFences(device_, 1, &fences_[frame_]); vkResetCommandBuffer(commandBuffers_[imageIndex], 0); record(imageIndex);
            VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT; VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO}; submit.waitSemaphoreCount=1; submit.pWaitSemaphores=&imageAvailable_[frame_]; submit.pWaitDstStageMask=&waitStage; submit.commandBufferCount=1; submit.pCommandBuffers=&commandBuffers_[imageIndex]; submit.signalSemaphoreCount=1; submit.pSignalSemaphores=&renderFinished_[frame_];
            VkResult submitted = vkQueueSubmit(graphicsQueue_, 1, &submit, fences_[frame_]); if (submitted != VK_SUCCESS) { LOGE("vkQueueSubmit: %d", submitted); break; }
            VkPresentInfoKHR present{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR}; present.waitSemaphoreCount=1; present.pWaitSemaphores=&renderFinished_[frame_]; present.swapchainCount=1; present.pSwapchains=&swapchain_; present.pImageIndices=&imageIndex;
            VkResult shown = vkQueuePresentKHR(presentQueue_, &present); if (shown == VK_ERROR_OUT_OF_DATE_KHR || shown == VK_SUBOPTIMAL_KHR || acquire == VK_SUBOPTIMAL_KHR) resizeRequested_.store(true); else if (shown != VK_SUCCESS) { LOGE("vkQueuePresentKHR: %d", shown); break; }
            frame_ = (frame_ + 1) % kFrames;
        }
    }
    void rebuild() {
        if (!device_) return; vkDeviceWaitIdle(device_); destroySwapchain();
        try { createSwapchain(); createRenderPass(); createFramebuffers(); createCommands(); status_ = "Vulkan active | " + deviceName_ + " | swapchain " + std::to_string(extent_.width) + "x" + std::to_string(extent_.height); }
        catch (const std::exception& e) { status_ = std::string("Vulkan resize failed: ") + e.what(); LOGE("%s", status_.c_str()); }
    }
    void destroySwapchain() {
        if (device_) { for (auto fb : framebuffers_) if (fb) vkDestroyFramebuffer(device_, fb, nullptr); framebuffers_.clear(); if (commandPool_) { vkDestroyCommandPool(device_, commandPool_, nullptr); commandPool_=VK_NULL_HANDLE; } commandBuffers_.clear(); if (renderPass_) vkDestroyRenderPass(device_, renderPass_, nullptr); renderPass_=VK_NULL_HANDLE; for (auto v : views_) if (v) vkDestroyImageView(device_, v, nullptr); views_.clear(); images_.clear(); if (swapchain_) vkDestroySwapchainKHR(device_, swapchain_, nullptr); swapchain_=VK_NULL_HANDLE; }
    }
    void cleanup() {
        if (device_) vkDeviceWaitIdle(device_);
        destroySwapchain();
        if (device_) { for (size_t i=0;i<kFrames;++i) { if (imageAvailable_[i]) vkDestroySemaphore(device_, imageAvailable_[i], nullptr); if (renderFinished_[i]) vkDestroySemaphore(device_, renderFinished_[i], nullptr); if (fences_[i]) vkDestroyFence(device_, fences_[i], nullptr); } vkDestroyDevice(device_, nullptr); device_=VK_NULL_HANDLE; }
        if (instance_ && surface_) { vkDestroySurfaceKHR(instance_, surface_, nullptr); surface_=VK_NULL_HANDLE; }
        if (instance_) { vkDestroyInstance(instance_, nullptr); instance_=VK_NULL_HANDLE; }
    }
};

static std::mutex gMutex;
static VulkanRenderer* gRenderer = nullptr;
}

extern "C" JNIEXPORT void JNICALL Java_com_neo3d_engine_MainActivity_nativeStart(JNIEnv* env, jobject, jobject surface) {
    std::lock_guard<std::mutex> lock(neo3d::gMutex);
    if (neo3d::gRenderer) { neo3d::gRenderer->stop(); delete neo3d::gRenderer; neo3d::gRenderer = nullptr; }
    ANativeWindow* window = ANativeWindow_fromSurface(env, surface);
    if (!window) { LOGE("ANativeWindow_fromSurface returned null"); return; }
    neo3d::gRenderer = new neo3d::VulkanRenderer(window); ANativeWindow_release(window); neo3d::gRenderer->start();
}
extern "C" JNIEXPORT void JNICALL Java_com_neo3d_engine_MainActivity_nativeResize(JNIEnv*, jobject, jint width, jint height) { std::lock_guard<std::mutex> lock(neo3d::gMutex); if (neo3d::gRenderer) neo3d::gRenderer->resize(width,height); }
extern "C" JNIEXPORT void JNICALL Java_com_neo3d_engine_MainActivity_nativeStop(JNIEnv*, jobject) { std::lock_guard<std::mutex> lock(neo3d::gMutex); if (neo3d::gRenderer) { neo3d::gRenderer->stop(); delete neo3d::gRenderer; neo3d::gRenderer=nullptr; } }
extern "C" JNIEXPORT jstring JNICALL Java_com_neo3d_engine_MainActivity_nativeStatus(JNIEnv* env, jobject) { std::lock_guard<std::mutex> lock(neo3d::gMutex); std::string s = neo3d::gRenderer ? neo3d::gRenderer->status() : "Vulkan renderer stopped"; return env->NewStringUTF(s.c_str()); }
