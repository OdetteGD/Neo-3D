#include <jni.h>
#include <android/native_window.h>
#include <android/native_window_jni.h>
#include <android/log.h>
#include <vulkan/vulkan.h>
#include "shader_blobs.hpp"
#include "scene/scene.hpp"
#include "render/forward_plus.hpp"
#include "assets/glb_reader.hpp"
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
    if (r != VK_SUCCESS) { LOGE("%s failed: VkResult=%d", op, static_cast<int>(r)); throw std::runtime_error(op); }
}
struct Vertex { float position[3]; float normal[3]; };
struct Mat4 { float v[16]{}; };
static Mat4 identity() { Mat4 m{}; m.v[0]=m.v[5]=m.v[10]=m.v[15]=1.0f; return m; }
static Mat4 multiply(const Mat4& a,const Mat4& b) {
    Mat4 o{};
    for(int c=0;c<4;++c) for(int r=0;r<4;++r)
        for(int k=0;k<4;++k) o.v[c*4+r]+=a.v[k*4+r]*b.v[c*4+k];
    return o;
}
static Mat4 perspective(float aspect) {
    const float f=1.0f/std::tan(0.78539816339f*0.5f), nearP=0.1f, farP=100.0f;
    Mat4 m{}; m.v[0]=f/std::max(aspect,0.01f); m.v[5]=-f;
    m.v[10]=farP/(nearP-farP); m.v[11]=-1.0f; m.v[14]=(farP*nearP)/(nearP-farP);
    return m;
}
static Mat4 translate(float x,float y,float z) { Mat4 m=identity(); m.v[12]=x; m.v[13]=y; m.v[14]=z; return m; }
static Mat4 rotateY(float a) { Mat4 m=identity(); float c=std::cos(a),s=std::sin(a); m.v[0]=c;m.v[2]=-s;m.v[8]=s;m.v[10]=c;return m; }
static Mat4 rotateX(float a) { Mat4 m=identity(); float c=std::cos(a),s=std::sin(a);m.v[5]=c;m.v[6]=s;m.v[9]=-s;m.v[10]=c;return m; }

class VulkanRenderer {
public:
    explicit VulkanRenderer(ANativeWindow* w):window_(w) { if(window_) ANativeWindow_acquire(window_); }
    ~VulkanRenderer(){ stop(); if(window_) ANativeWindow_release(window_); }
    void start() {
        try { initialize(); running_.store(true); started_=std::chrono::steady_clock::now(); thread_=std::thread(&VulkanRenderer::renderLoop,this); }
        catch(const std::exception& e){ status_=std::string("Vulkan initialization failed: ")+e.what();LOGE("%s",status_.c_str());cleanup(); }
    }
    void resize(int,int){ resizeRequested_.store(true); }
    void stop(){ running_.store(false); if(thread_.joinable()) thread_.join(); cleanup(); }
    std::string status() const { return status_; }
private:
    ANativeWindow* window_=nullptr;
    VkInstance instance_=VK_NULL_HANDLE; VkSurfaceKHR surface_=VK_NULL_HANDLE;
    VkPhysicalDevice physical_=VK_NULL_HANDLE; VkDevice device_=VK_NULL_HANDLE;
    uint32_t graphicsFamily_=UINT32_MAX,presentFamily_=UINT32_MAX;
    VkQueue graphicsQueue_=VK_NULL_HANDLE,presentQueue_=VK_NULL_HANDLE;
    VkSwapchainKHR swapchain_=VK_NULL_HANDLE; VkFormat format_=VK_FORMAT_UNDEFINED,depthFormat_=VK_FORMAT_D32_SFLOAT;
    VkExtent2D extent_{};
    std::vector<VkImage> images_; std::vector<VkImageView> views_;
    std::vector<VkImage> depthImages_; std::vector<VkDeviceMemory> depthMemory_; std::vector<VkImageView> depthViews_;
    VkRenderPass renderPass_=VK_NULL_HANDLE; std::vector<VkFramebuffer> framebuffers_;
    VkCommandPool commandPool_=VK_NULL_HANDLE; std::vector<VkCommandBuffer> commandBuffers_;
    VkPipelineLayout pipelineLayout_=VK_NULL_HANDLE; VkPipeline pipeline_=VK_NULL_HANDLE;
    VkBuffer vertexBuffer_=VK_NULL_HANDLE,indexBuffer_=VK_NULL_HANDLE;
    VkDeviceMemory vertexMemory_=VK_NULL_HANDLE,indexMemory_=VK_NULL_HANDLE;
    uint32_t indexCount_=0;
    static constexpr size_t kFrames=2;
    std::array<VkSemaphore,kFrames> imageAvailable_{}; std::array<VkSemaphore,kFrames> renderFinished_{}; std::array<VkFence,kFrames> fences_{};
    size_t frame_=0; std::atomic<bool> running_{false},resizeRequested_{false}; std::thread thread_;
    std::chrono::steady_clock::time_point started_{};
    std::string deviceName_="unknown GPU",status_="Vulkan renderer not initialized";
    std::mutex lifecycleMutex_;

    void initialize(){
        VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};app.pApplicationName="Neo-3D";app.applicationVersion=VK_MAKE_VERSION(0,2,0);app.pEngineName="Neo-3D Mobile Vulkan Engine";app.engineVersion=VK_MAKE_VERSION(0,2,0);app.apiVersion=VK_API_VERSION_1_0;
        const char* exts[]={VK_KHR_SURFACE_EXTENSION_NAME,VK_KHR_ANDROID_SURFACE_EXTENSION_NAME};
        VkInstanceCreateInfo ci{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};ci.pApplicationInfo=&app;ci.enabledExtensionCount=2;ci.ppEnabledExtensionNames=exts;
        check(vkCreateInstance(&ci,nullptr,&instance_),"vkCreateInstance");
        VkAndroidSurfaceCreateInfoKHR si{VK_STRUCTURE_TYPE_ANDROID_SURFACE_CREATE_INFO_KHR};si.window=window_;
        check(vkCreateAndroidSurfaceKHR(instance_,&si,nullptr,&surface_),"vkCreateAndroidSurfaceKHR");
        selectDevice();createDevice();createGeometry();createSwapchain();chooseDepthFormat();createRenderPass();createDepthResources();createFramebuffers();createCommands();createGraphicsPipeline();createSync();
        status_="Vulkan 3D | "+deviceName_+" | "+std::to_string(extent_.width)+"x"+std::to_string(extent_.height)+" | indexed cube + depth + GGX PBR";
        LOGI("%s",status_.c_str());
    }
    void selectDevice(){
        uint32_t count=0;check(vkEnumeratePhysicalDevices(instance_,&count,nullptr),"vkEnumeratePhysicalDevices(count)");
        if(!count)throw std::runtime_error("No Vulkan GPU available");
        std::vector<VkPhysicalDevice> ds(count);check(vkEnumeratePhysicalDevices(instance_,&count,ds.data()),"vkEnumeratePhysicalDevices");
        for(auto d:ds){
            uint32_t qc=0;vkGetPhysicalDeviceQueueFamilyProperties(d,&qc,nullptr);std::vector<VkQueueFamilyProperties> qs(qc);vkGetPhysicalDeviceQueueFamilyProperties(d,&qc,qs.data());
            uint32_t gf=UINT32_MAX,pf=UINT32_MAX;
            for(uint32_t i=0;i<qc;++i){if(qs[i].queueFlags&VK_QUEUE_GRAPHICS_BIT)gf=i;VkBool32 supported=VK_FALSE;vkGetPhysicalDeviceSurfaceSupportKHR(d,i,surface_,&supported);if(supported)pf=i;}
            if(gf==UINT32_MAX||pf==UINT32_MAX)continue;
            uint32_t ec=0;vkEnumerateDeviceExtensionProperties(d,nullptr,&ec,nullptr);std::vector<VkExtensionProperties> es(ec);vkEnumerateDeviceExtensionProperties(d,nullptr,&ec,es.data());
            if(std::none_of(es.begin(),es.end(),[](const auto&e){return std::strcmp(e.extensionName,VK_KHR_SWAPCHAIN_EXTENSION_NAME)==0;}))continue;
            physical_=d;graphicsFamily_=gf;presentFamily_=pf;VkPhysicalDeviceProperties p{};vkGetPhysicalDeviceProperties(d,&p);deviceName_=p.deviceName;return;
        }
        throw std::runtime_error("No physical device supports graphics and Android swapchain presentation");
    }
    void createDevice(){
        float priority=1.0f;std::vector<VkDeviceQueueCreateInfo> qs;
        for(uint32_t family:{graphicsFamily_,presentFamily_}){if(std::any_of(qs.begin(),qs.end(),[family](const auto&q){return q.queueFamilyIndex==family;}))continue;VkDeviceQueueCreateInfo q{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};q.queueFamilyIndex=family;q.queueCount=1;q.pQueuePriorities=&priority;qs.push_back(q);}
        VkPhysicalDeviceFeatures features{};VkDeviceCreateInfo ci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};ci.queueCreateInfoCount=static_cast<uint32_t>(qs.size());ci.pQueueCreateInfos=qs.data();const char* ext=VK_KHR_SWAPCHAIN_EXTENSION_NAME;ci.enabledExtensionCount=1;ci.ppEnabledExtensionNames=&ext;ci.pEnabledFeatures=&features;
        check(vkCreateDevice(physical_,&ci,nullptr,&device_),"vkCreateDevice");vkGetDeviceQueue(device_,graphicsFamily_,0,&graphicsQueue_);vkGetDeviceQueue(device_,presentFamily_,0,&presentQueue_);
    }
    uint32_t memoryType(uint32_t bits,VkMemoryPropertyFlags flags){
        VkPhysicalDeviceMemoryProperties p{};vkGetPhysicalDeviceMemoryProperties(physical_,&p);
        for(uint32_t i=0;i<p.memoryTypeCount;++i)if((bits&(1u<<i))&&(p.memoryTypes[i].propertyFlags&flags)==flags)return i;
        throw std::runtime_error("No Vulkan memory type matches requested properties");
    }
    void createBuffer(VkDeviceSize size,VkBufferUsageFlags usage,VkBuffer& buffer,VkDeviceMemory& memory,const void* data){
        VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};bi.size=size;bi.usage=usage;bi.sharingMode=VK_SHARING_MODE_EXCLUSIVE;check(vkCreateBuffer(device_,&bi,nullptr,&buffer),"vkCreateBuffer");
        VkMemoryRequirements req{};vkGetBufferMemoryRequirements(device_,buffer,&req);VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};ai.allocationSize=req.size;ai.memoryTypeIndex=memoryType(req.memoryTypeBits,VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);check(vkAllocateMemory(device_,&ai,nullptr,&memory),"vkAllocateMemory(buffer)");check(vkBindBufferMemory(device_,buffer,memory,0),"vkBindBufferMemory");
        if(data){void* mapped=nullptr;check(vkMapMemory(device_,memory,0,size,0,&mapped),"vkMapMemory");std::memcpy(mapped,data,static_cast<size_t>(size));vkUnmapMemory(device_,memory);}
    }
    void createGeometry(){
        // Six independently-normaled faces: 24 vertices and 36 indices for a correctly lit cube.
        const std::array<Vertex,24> v={{
            {{-1,-1, 1},{0,0,1}},{{1,-1,1},{0,0,1}},{{1,1,1},{0,0,1}},{{-1,1,1},{0,0,1}},
            {{1,-1,-1},{0,0,-1}},{{-1,-1,-1},{0,0,-1}},{{-1,1,-1},{0,0,-1}},{{1,1,-1},{0,0,-1}},
            {{-1,-1,-1},{-1,0,0}},{{-1,-1,1},{-1,0,0}},{{-1,1,1},{-1,0,0}},{{-1,1,-1},{-1,0,0}},
            {{1,-1,1},{1,0,0}},{{1,-1,-1},{1,0,0}},{{1,1,-1},{1,0,0}},{{1,1,1},{1,0,0}},
            {{-1,1,1},{0,1,0}},{{1,1,1},{0,1,0}},{{1,1,-1},{0,1,0}},{{-1,1,-1},{0,1,0}},
            {{-1,-1,-1},{0,-1,0}},{{1,-1,-1},{0,-1,0}},{{1,-1,1},{0,-1,0}},{{-1,-1,1},{0,-1,0}}
        }};
        const std::array<uint32_t,36> idx={{0,1,2,2,3,0,4,5,6,6,7,4,8,9,10,10,11,8,12,13,14,14,15,12,16,17,18,18,19,16,20,21,22,22,23,20}};
        indexCount_=static_cast<uint32_t>(idx.size());createBuffer(sizeof(v),VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,vertexBuffer_,vertexMemory_,v.data());createBuffer(sizeof(idx),VK_BUFFER_USAGE_INDEX_BUFFER_BIT,indexBuffer_,indexMemory_,idx.data());
    }
    void createSwapchain(){
        VkSurfaceCapabilitiesKHR caps{};check(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physical_,surface_,&caps),"vkGetPhysicalDeviceSurfaceCapabilitiesKHR");
        uint32_t n=0;vkGetPhysicalDeviceSurfaceFormatsKHR(physical_,surface_,&n,nullptr);if(!n)throw std::runtime_error("No swapchain surface formats");std::vector<VkSurfaceFormatKHR> fs(n);vkGetPhysicalDeviceSurfaceFormatsKHR(physical_,surface_,&n,fs.data());
        VkSurfaceFormatKHR chosen=fs.front();for(const auto& f:fs)if(f.format==VK_FORMAT_B8G8R8A8_UNORM&&f.colorSpace==VK_COLOR_SPACE_SRGB_NONLINEAR_KHR){chosen=f;break;}format_=chosen.format;
        if(caps.currentExtent.width!=UINT32_MAX)extent_=caps.currentExtent;else{extent_.width=std::clamp(static_cast<uint32_t>(ANativeWindow_getWidth(window_)),caps.minImageExtent.width,caps.maxImageExtent.width);extent_.height=std::clamp(static_cast<uint32_t>(ANativeWindow_getHeight(window_)),caps.minImageExtent.height,caps.maxImageExtent.height);}
        uint32_t ic=caps.minImageCount+1;if(caps.maxImageCount&&ic>caps.maxImageCount)ic=caps.maxImageCount;
        VkSwapchainCreateInfoKHR ci{VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR};ci.surface=surface_;ci.minImageCount=ic;ci.imageFormat=chosen.format;ci.imageColorSpace=chosen.colorSpace;ci.imageExtent=extent_;ci.imageArrayLayers=1;ci.imageUsage=VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
        uint32_t families[]={graphicsFamily_,presentFamily_};if(graphicsFamily_!=presentFamily_){ci.imageSharingMode=VK_SHARING_MODE_CONCURRENT;ci.queueFamilyIndexCount=2;ci.pQueueFamilyIndices=families;}else ci.imageSharingMode=VK_SHARING_MODE_EXCLUSIVE;
        ci.preTransform=caps.currentTransform;ci.compositeAlpha=VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;ci.presentMode=VK_PRESENT_MODE_FIFO_KHR;ci.clipped=VK_TRUE;
        check(vkCreateSwapchainKHR(device_,&ci,nullptr,&swapchain_),"vkCreateSwapchainKHR");uint32_t actual=0;vkGetSwapchainImagesKHR(device_,swapchain_,&actual,nullptr);images_.resize(actual);check(vkGetSwapchainImagesKHR(device_,swapchain_,&actual,images_.data()),"vkGetSwapchainImagesKHR");
        views_.resize(images_.size());for(size_t i=0;i<images_.size();++i){VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};vi.image=images_[i];vi.viewType=VK_IMAGE_VIEW_TYPE_2D;vi.format=format_;vi.components={VK_COMPONENT_SWIZZLE_IDENTITY,VK_COMPONENT_SWIZZLE_IDENTITY,VK_COMPONENT_SWIZZLE_IDENTITY,VK_COMPONENT_SWIZZLE_IDENTITY};vi.subresourceRange={VK_IMAGE_ASPECT_COLOR_BIT,0,1,0,1};check(vkCreateImageView(device_,&vi,nullptr,&views_[i]),"vkCreateImageView(swapchain)");}
    }
    void chooseDepthFormat(){
        for(VkFormat f:{VK_FORMAT_D32_SFLOAT,VK_FORMAT_D24_UNORM_S8_UINT,VK_FORMAT_D16_UNORM}){VkFormatProperties p{};vkGetPhysicalDeviceFormatProperties(physical_,f,&p);if(p.optimalTilingFeatures&VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT){depthFormat_=f;return;}}
        throw std::runtime_error("GPU has no supported depth attachment format");
    }
    void createRenderPass(){
        std::array<VkAttachmentDescription,2> a{};a[0].format=format_;a[0].samples=VK_SAMPLE_COUNT_1_BIT;a[0].loadOp=VK_ATTACHMENT_LOAD_OP_CLEAR;a[0].storeOp=VK_ATTACHMENT_STORE_OP_STORE;a[0].stencilLoadOp=VK_ATTACHMENT_LOAD_OP_DONT_CARE;a[0].stencilStoreOp=VK_ATTACHMENT_STORE_OP_DONT_CARE;a[0].initialLayout=VK_IMAGE_LAYOUT_UNDEFINED;a[0].finalLayout=VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
        a[1].format=depthFormat_;a[1].samples=VK_SAMPLE_COUNT_1_BIT;a[1].loadOp=VK_ATTACHMENT_LOAD_OP_CLEAR;a[1].storeOp=VK_ATTACHMENT_STORE_OP_DONT_CARE;a[1].stencilLoadOp=VK_ATTACHMENT_LOAD_OP_DONT_CARE;a[1].stencilStoreOp=VK_ATTACHMENT_STORE_OP_DONT_CARE;a[1].initialLayout=VK_IMAGE_LAYOUT_UNDEFINED;a[1].finalLayout=VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        VkAttachmentReference color{0,VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL},depth{1,VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};VkSubpassDescription sub{};sub.pipelineBindPoint=VK_PIPELINE_BIND_POINT_GRAPHICS;sub.colorAttachmentCount=1;sub.pColorAttachments=&color;sub.pDepthStencilAttachment=&depth;
        VkSubpassDependency dep{};dep.srcSubpass=VK_SUBPASS_EXTERNAL;dep.dstSubpass=0;dep.srcStageMask=VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT|VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;dep.dstStageMask=dep.srcStageMask;dep.dstAccessMask=VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT|VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        VkRenderPassCreateInfo ci{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};ci.attachmentCount=2;ci.pAttachments=a.data();ci.subpassCount=1;ci.pSubpasses=&sub;ci.dependencyCount=1;ci.pDependencies=&dep;check(vkCreateRenderPass(device_,&ci,nullptr,&renderPass_),"vkCreateRenderPass");
    }
    void createDepthResources(){
        depthImages_.resize(images_.size());depthMemory_.resize(images_.size());depthViews_.resize(images_.size());
        for(size_t i=0;i<images_.size();++i){VkImageCreateInfo ii{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};ii.imageType=VK_IMAGE_TYPE_2D;ii.extent={extent_.width,extent_.height,1};ii.mipLevels=1;ii.arrayLayers=1;ii.format=depthFormat_;ii.tiling=VK_IMAGE_TILING_OPTIMAL;ii.initialLayout=VK_IMAGE_LAYOUT_UNDEFINED;ii.usage=VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;ii.samples=VK_SAMPLE_COUNT_1_BIT;ii.sharingMode=VK_SHARING_MODE_EXCLUSIVE;check(vkCreateImage(device_,&ii,nullptr,&depthImages_[i]),"vkCreateImage(depth)");
            VkMemoryRequirements req{};vkGetImageMemoryRequirements(device_,depthImages_[i],&req);VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};ai.allocationSize=req.size;ai.memoryTypeIndex=memoryType(req.memoryTypeBits,VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);check(vkAllocateMemory(device_,&ai,nullptr,&depthMemory_[i]),"vkAllocateMemory(depth)");check(vkBindImageMemory(device_,depthImages_[i],depthMemory_[i],0),"vkBindImageMemory(depth)");
            VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};vi.image=depthImages_[i];vi.viewType=VK_IMAGE_VIEW_TYPE_2D;vi.format=depthFormat_;vi.subresourceRange={VK_IMAGE_ASPECT_DEPTH_BIT,0,1,0,1};check(vkCreateImageView(device_,&vi,nullptr,&depthViews_[i]),"vkCreateImageView(depth)");
        }
    }
    VkShaderModule shader(const uint32_t* code,size_t bytes){VkShaderModuleCreateInfo ci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};ci.codeSize=bytes;ci.pCode=code;VkShaderModule m=VK_NULL_HANDLE;check(vkCreateShaderModule(device_,&ci,nullptr,&m),"vkCreateShaderModule");return m;}
    void createGraphicsPipeline(){
        VkShaderModule vert=shader(kMeshVert,kMeshVertSize),frag=shader(kMeshFrag,kMeshFragSize);
        try {
            VkPushConstantRange range{};range.stageFlags=VK_SHADER_STAGE_VERTEX_BIT;range.offset=0;range.size=sizeof(Mat4)*2;
            VkPipelineLayoutCreateInfo li{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};li.pushConstantRangeCount=1;li.pPushConstantRanges=&range;check(vkCreatePipelineLayout(device_,&li,nullptr,&pipelineLayout_),"vkCreatePipelineLayout");
            std::array<VkPipelineShaderStageCreateInfo,2> stages{};stages[0]={VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};stages[0].stage=VK_SHADER_STAGE_VERTEX_BIT;stages[0].module=vert;stages[0].pName="main";stages[1]={VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};stages[1].stage=VK_SHADER_STAGE_FRAGMENT_BIT;stages[1].module=frag;stages[1].pName="main";
            VkVertexInputBindingDescription binding{0,sizeof(Vertex),VK_VERTEX_INPUT_RATE_VERTEX};std::array<VkVertexInputAttributeDescription,2> attrs={VkVertexInputAttributeDescription{0,0,VK_FORMAT_R32G32B32_SFLOAT,offsetof(Vertex,position)},VkVertexInputAttributeDescription{1,0,VK_FORMAT_R32G32B32_SFLOAT,offsetof(Vertex,normal)}};
            VkPipelineVertexInputStateCreateInfo vi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};vi.vertexBindingDescriptionCount=1;vi.pVertexBindingDescriptions=&binding;vi.vertexAttributeDescriptionCount=static_cast<uint32_t>(attrs.size());vi.pVertexAttributeDescriptions=attrs.data();
            VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};ia.topology=VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
            VkViewport vp{0,0,static_cast<float>(extent_.width),static_cast<float>(extent_.height),0,1};VkRect2D sc{{0,0},extent_};VkPipelineViewportStateCreateInfo vsi{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};vsi.viewportCount=1;vsi.pViewports=&vp;vsi.scissorCount=1;vsi.pScissors=&sc;
            VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};rs.polygonMode=VK_POLYGON_MODE_FILL;rs.cullMode=VK_CULL_MODE_BACK_BIT;rs.frontFace=VK_FRONT_FACE_COUNTER_CLOCKWISE;rs.lineWidth=1.0f;
            VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};ms.rasterizationSamples=VK_SAMPLE_COUNT_1_BIT;
            VkPipelineDepthStencilStateCreateInfo ds{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};ds.depthTestEnable=VK_TRUE;ds.depthWriteEnable=VK_TRUE;ds.depthCompareOp=VK_COMPARE_OP_LESS;ds.minDepthBounds=0;ds.maxDepthBounds=1;
            VkPipelineColorBlendAttachmentState ba{};ba.colorWriteMask=VK_COLOR_COMPONENT_R_BIT|VK_COLOR_COMPONENT_G_BIT|VK_COLOR_COMPONENT_B_BIT|VK_COLOR_COMPONENT_A_BIT;
            VkPipelineColorBlendStateCreateInfo bs{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};bs.attachmentCount=1;bs.pAttachments=&ba;
            VkGraphicsPipelineCreateInfo pi{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};pi.stageCount=2;pi.pStages=stages.data();pi.pVertexInputState=&vi;pi.pInputAssemblyState=&ia;pi.pViewportState=&vsi;pi.pRasterizationState=&rs;pi.pMultisampleState=&ms;pi.pDepthStencilState=&ds;pi.pColorBlendState=&bs;pi.layout=pipelineLayout_;pi.renderPass=renderPass_;pi.subpass=0;
            check(vkCreateGraphicsPipelines(device_,VK_NULL_HANDLE,1,&pi,nullptr,&pipeline_),"vkCreateGraphicsPipelines");
        } catch(...) {vkDestroyShaderModule(device_,vert,nullptr);vkDestroyShaderModule(device_,frag,nullptr);throw;}
        vkDestroyShaderModule(device_,vert,nullptr);vkDestroyShaderModule(device_,frag,nullptr);
    }
    void createFramebuffers(){
        framebuffers_.resize(views_.size());for(size_t i=0;i<views_.size();++i){VkImageView atts[]={views_[i],depthViews_[i]};VkFramebufferCreateInfo ci{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};ci.renderPass=renderPass_;ci.attachmentCount=2;ci.pAttachments=atts;ci.width=extent_.width;ci.height=extent_.height;ci.layers=1;check(vkCreateFramebuffer(device_,&ci,nullptr,&framebuffers_[i]),"vkCreateFramebuffer");}
    }
    void createCommands(){VkCommandPoolCreateInfo pi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};pi.flags=VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;pi.queueFamilyIndex=graphicsFamily_;check(vkCreateCommandPool(device_,&pi,nullptr,&commandPool_),"vkCreateCommandPool");commandBuffers_.resize(framebuffers_.size());VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};ai.commandPool=commandPool_;ai.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY;ai.commandBufferCount=static_cast<uint32_t>(commandBuffers_.size());check(vkAllocateCommandBuffers(device_,&ai,commandBuffers_.data()),"vkAllocateCommandBuffers");}
    void createSync(){VkSemaphoreCreateInfo si{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};VkFenceCreateInfo fi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};fi.flags=VK_FENCE_CREATE_SIGNALED_BIT;for(size_t i=0;i<kFrames;++i){check(vkCreateSemaphore(device_,&si,nullptr,&imageAvailable_[i]),"vkCreateSemaphore");check(vkCreateSemaphore(device_,&si,nullptr,&renderFinished_[i]),"vkCreateSemaphore");check(vkCreateFence(device_,&fi,nullptr,&fences_[i]),"vkCreateFence");}}
    void record(uint32_t i){
        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};check(vkBeginCommandBuffer(commandBuffers_[i],&bi),"vkBeginCommandBuffer");
        VkClearValue clears[2]{};clears[0].color={{0.025f,0.045f,0.075f,1.0f}};clears[1].depthStencil={1.0f,0};
        VkRenderPassBeginInfo rp{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};rp.renderPass=renderPass_;rp.framebuffer=framebuffers_[i];rp.renderArea={{0,0},extent_};rp.clearValueCount=2;rp.pClearValues=clears;
        vkCmdBeginRenderPass(commandBuffers_[i],&rp,VK_SUBPASS_CONTENTS_INLINE);vkCmdBindPipeline(commandBuffers_[i],VK_PIPELINE_BIND_POINT_GRAPHICS,pipeline_);
        VkDeviceSize offset=0;vkCmdBindVertexBuffers(commandBuffers_[i],0,1,&vertexBuffer_,&offset);vkCmdBindIndexBuffer(commandBuffers_[i],indexBuffer_,0,VK_INDEX_TYPE_UINT32);
        float t=std::chrono::duration<float>(std::chrono::steady_clock::now()-started_).count();
        Mat4 model=multiply(rotateY(t*0.65f),rotateX(-0.28f));Mat4 view=translate(0,0,-4.4f);Mat4 mvp=multiply(perspective(static_cast<float>(extent_.width)/static_cast<float>(std::max(1u,extent_.height))),multiply(view,model));
        struct Push{Mat4 mvp;Mat4 model;} push{mvp,model};vkCmdPushConstants(commandBuffers_[i],pipelineLayout_,VK_SHADER_STAGE_VERTEX_BIT,0,sizeof(push),&push);
        vkCmdDrawIndexed(commandBuffers_[i],indexCount_,1,0,0,0);vkCmdEndRenderPass(commandBuffers_[i]);check(vkEndCommandBuffer(commandBuffers_[i]),"vkEndCommandBuffer");
    }
    void renderLoop(){
        while(running_.load()){
            if(resizeRequested_.exchange(false)){std::lock_guard<std::mutex> lock(lifecycleMutex_);rebuild();}
            if(!swapchain_||!device_||!pipeline_)break;
            check(vkWaitForFences(device_,1,&fences_[frame_],VK_TRUE,UINT64_MAX),"vkWaitForFences");
            uint32_t imageIndex=0;VkResult ac=vkAcquireNextImageKHR(device_,swapchain_,UINT64_MAX,imageAvailable_[frame_],VK_NULL_HANDLE,&imageIndex);
            if(ac==VK_ERROR_OUT_OF_DATE_KHR){resizeRequested_.store(true);continue;}if(ac!=VK_SUCCESS&&ac!=VK_SUBOPTIMAL_KHR){LOGE("vkAcquireNextImageKHR=%d",ac);break;}
            vkResetFences(device_,1,&fences_[frame_]);vkResetCommandBuffer(commandBuffers_[imageIndex],0);record(imageIndex);
            VkPipelineStageFlags wait=VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};si.waitSemaphoreCount=1;si.pWaitSemaphores=&imageAvailable_[frame_];si.pWaitDstStageMask=&wait;si.commandBufferCount=1;si.pCommandBuffers=&commandBuffers_[imageIndex];si.signalSemaphoreCount=1;si.pSignalSemaphores=&renderFinished_[frame_];
            VkResult sub=vkQueueSubmit(graphicsQueue_,1,&si,fences_[frame_]);if(sub!=VK_SUCCESS){LOGE("vkQueueSubmit=%d",sub);break;}
            VkPresentInfoKHR pi{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};pi.waitSemaphoreCount=1;pi.pWaitSemaphores=&renderFinished_[frame_];pi.swapchainCount=1;pi.pSwapchains=&swapchain_;pi.pImageIndices=&imageIndex;VkResult pr=vkQueuePresentKHR(presentQueue_,&pi);
            if(pr==VK_ERROR_OUT_OF_DATE_KHR||pr==VK_SUBOPTIMAL_KHR||ac==VK_SUBOPTIMAL_KHR)resizeRequested_.store(true);else if(pr!=VK_SUCCESS){LOGE("vkQueuePresentKHR=%d",pr);break;}frame_=(frame_+1)%kFrames;
        }
    }
    void rebuild(){
        if(!device_)return;vkDeviceWaitIdle(device_);destroySwapchain();
        try{createSwapchain();chooseDepthFormat();createRenderPass();createDepthResources();createFramebuffers();createCommands();createGraphicsPipeline();status_="Vulkan 3D | "+deviceName_+" | "+std::to_string(extent_.width)+"x"+std::to_string(extent_.height)+" | indexed cube + depth + GGX PBR";}
        catch(const std::exception&e){status_=std::string("Swapchain rebuild failed: ")+e.what();LOGE("%s",status_.c_str());}
    }
    void destroySwapchain(){
        if(!device_)return;
        if(pipeline_)vkDestroyPipeline(device_,pipeline_,nullptr);pipeline_=VK_NULL_HANDLE;
        if(pipelineLayout_)vkDestroyPipelineLayout(device_,pipelineLayout_,nullptr);pipelineLayout_=VK_NULL_HANDLE;
        for(auto f:framebuffers_)if(f)vkDestroyFramebuffer(device_,f,nullptr);framebuffers_.clear();
        if(commandPool_)vkDestroyCommandPool(device_,commandPool_,nullptr);commandPool_=VK_NULL_HANDLE;commandBuffers_.clear();
        if(renderPass_)vkDestroyRenderPass(device_,renderPass_,nullptr);renderPass_=VK_NULL_HANDLE;
        for(auto v:depthViews_)if(v)vkDestroyImageView(device_,v,nullptr);depthViews_.clear();
        for(auto im:depthImages_)if(im)vkDestroyImage(device_,im,nullptr);depthImages_.clear();
        for(auto m:depthMemory_)if(m)vkFreeMemory(device_,m,nullptr);depthMemory_.clear();
        for(auto v:views_)if(v)vkDestroyImageView(device_,v,nullptr);views_.clear();images_.clear();
        if(swapchain_)vkDestroySwapchainKHR(device_,swapchain_,nullptr);swapchain_=VK_NULL_HANDLE;
    }
    void cleanup(){
        if(device_)vkDeviceWaitIdle(device_);destroySwapchain();
        if(device_){if(vertexBuffer_)vkDestroyBuffer(device_,vertexBuffer_,nullptr);if(indexBuffer_)vkDestroyBuffer(device_,indexBuffer_,nullptr);if(vertexMemory_)vkFreeMemory(device_,vertexMemory_,nullptr);if(indexMemory_)vkFreeMemory(device_,indexMemory_,nullptr);
            for(size_t i=0;i<kFrames;++i){if(imageAvailable_[i])vkDestroySemaphore(device_,imageAvailable_[i],nullptr);if(renderFinished_[i])vkDestroySemaphore(device_,renderFinished_[i],nullptr);if(fences_[i])vkDestroyFence(device_,fences_[i],nullptr);}vkDestroyDevice(device_,nullptr);device_=VK_NULL_HANDLE;}
        if(instance_&&surface_){vkDestroySurfaceKHR(instance_,surface_,nullptr);surface_=VK_NULL_HANDLE;}if(instance_){vkDestroyInstance(instance_,nullptr);instance_=VK_NULL_HANDLE;}
    }
};
static std::mutex gMutex;static VulkanRenderer* gRenderer=nullptr;
}
extern "C" JNIEXPORT void JNICALL Java_com_neo3d_engine_MainActivity_nativeStart(JNIEnv* env,jobject,jobject surface){
    std::lock_guard<std::mutex> lock(neo3d::gMutex);if(neo3d::gRenderer){neo3d::gRenderer->stop();delete neo3d::gRenderer;neo3d::gRenderer=nullptr;}
    ANativeWindow*w=ANativeWindow_fromSurface(env,surface);if(!w){LOGE("ANativeWindow_fromSurface returned null");return;}neo3d::gRenderer=new neo3d::VulkanRenderer(w);ANativeWindow_release(w);neo3d::gRenderer->start();
}
extern "C" JNIEXPORT void JNICALL Java_com_neo3d_engine_MainActivity_nativeResize(JNIEnv*,jobject,jint w,jint h){std::lock_guard<std::mutex> lock(neo3d::gMutex);if(neo3d::gRenderer)neo3d::gRenderer->resize(w,h);}
extern "C" JNIEXPORT void JNICALL Java_com_neo3d_engine_MainActivity_nativeStop(JNIEnv*,jobject){std::lock_guard<std::mutex> lock(neo3d::gMutex);if(neo3d::gRenderer){neo3d::gRenderer->stop();delete neo3d::gRenderer;neo3d::gRenderer=nullptr;}}
extern "C" JNIEXPORT jstring JNICALL Java_com_neo3d_engine_MainActivity_nativeStatus(JNIEnv* env,jobject){std::lock_guard<std::mutex> lock(neo3d::gMutex);std::string s=neo3d::gRenderer?neo3d::gRenderer->status():"Vulkan renderer stopped";return env->NewStringUTF(s.c_str());}
