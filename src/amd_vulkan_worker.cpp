#include "amd_vulkan_worker.h"

#include <fstream>
#include <iostream>
#include <vector>

#include <vulkan/vulkan.h>

#ifndef AMD_VULKAN_SHADER_PATH
#define AMD_VULKAN_SHADER_PATH "build/amd_heat.spv"
#endif

namespace {

constexpr uint32_t AMD_VENDOR_ID = 0x1002;
constexpr uint32_t WORK_ITEM_COUNT = 1024 * 256;

bool checkVulkan( VkResult result, const char* operation ) {
    if ( result == VK_SUCCESS ) {
        return true;
    }

    std::cerr << "Vulkan AMD backend failed during " << operation << ": " << result << std::endl;
    return false;
}

VkInstance createInstance() {
    VkApplicationInfo applicationInfo{};
    applicationInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    applicationInfo.pApplicationName = "busywork";
    applicationInfo.applicationVersion = VK_MAKE_VERSION( 1, 0, 0 );
    applicationInfo.pEngineName = "busywork";
    applicationInfo.engineVersion = VK_MAKE_VERSION( 1, 0, 0 );
    applicationInfo.apiVersion = VK_API_VERSION_1_0;

    VkInstanceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pApplicationInfo = &applicationInfo;

    VkInstance instance = VK_NULL_HANDLE;
    if ( !checkVulkan( vkCreateInstance( &createInfo, nullptr, &instance ), "instance creation" ) ) {
        return VK_NULL_HANDLE;
    }

    return instance;
}

int findComputeQueueFamily( VkPhysicalDevice device ) {
    uint32_t count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties( device, &count, nullptr );
    std::vector<VkQueueFamilyProperties> families( count );
    vkGetPhysicalDeviceQueueFamilyProperties( device, &count, families.data() );

    for ( uint32_t index = 0; index < count; ++index ) {
        if ( families[index].queueFlags & VK_QUEUE_COMPUTE_BIT ) {
            return static_cast<int>( index );
        }
    }

    return -1;
}

uint32_t findMemoryType( VkPhysicalDevice device, uint32_t bits, VkMemoryPropertyFlags requiredProperties ) {
    VkPhysicalDeviceMemoryProperties properties{};
    vkGetPhysicalDeviceMemoryProperties( device, &properties );

    for ( uint32_t index = 0; index < properties.memoryTypeCount; ++index ) {
        if ( ( bits & ( 1u << index ) ) && ( properties.memoryTypes[index].propertyFlags & requiredProperties ) == requiredProperties ) {
            return index;
        }
    }

    return UINT32_MAX;
}

VkShaderModule loadShaderModule( VkDevice device ) {
    std::ifstream file( AMD_VULKAN_SHADER_PATH, std::ios::binary | std::ios::ate );
    if ( !file.is_open() ) {
        std::cerr << "Vulkan AMD backend could not open shader: " << AMD_VULKAN_SHADER_PATH << std::endl;
        return VK_NULL_HANDLE;
    }

    std::streamsize size = file.tellg();
    if ( size <= 0 || size % static_cast<std::streamsize>( sizeof( uint32_t ) ) != 0 ) {
        std::cerr << "Vulkan AMD backend found an invalid SPIR-V shader" << std::endl;
        return VK_NULL_HANDLE;
    }

    std::vector<uint32_t> code( static_cast<size_t>( size ) / sizeof( uint32_t ) );
    file.seekg( 0 );
    if ( !file.read( reinterpret_cast<char*>( code.data() ), size ) ) {
        std::cerr << "Vulkan AMD backend could not read shader" << std::endl;
        return VK_NULL_HANDLE;
    }

    VkShaderModuleCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.codeSize = static_cast<size_t>( size );
    createInfo.pCode = code.data();

    VkShaderModule shader = VK_NULL_HANDLE;
    if ( !checkVulkan( vkCreateShaderModule( device, &createInfo, nullptr, &shader ), "shader module creation" ) ) {
        return VK_NULL_HANDLE;
    }

    return shader;
}

}  // namespace

struct AmdVulkanContext {
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    uint32_t queueFamilyIndex = 0;
    VkBuffer outputBuffer = VK_NULL_HANDLE;
    VkDeviceMemory outputMemory = VK_NULL_HANDLE;
    VkDescriptorSetLayout descriptorSetLayout = VK_NULL_HANDLE;
    VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
    VkDescriptorSet descriptorSet = VK_NULL_HANDLE;
    VkShaderModule shader = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkCommandPool commandPool = VK_NULL_HANDLE;
    VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
};

std::vector<AmdVulkanDevice> enumerateAmdVulkanDevices() {
    std::vector<AmdVulkanDevice> devices;
    VkInstance instance = createInstance();
    if ( instance == VK_NULL_HANDLE ) {
        return devices;
    }

    uint32_t count = 0;
    if ( !checkVulkan( vkEnumeratePhysicalDevices( instance, &count, nullptr ), "physical-device enumeration" ) || count == 0 ) {
        vkDestroyInstance( instance, nullptr );
        return devices;
    }

    std::vector<VkPhysicalDevice> physicalDevices( count );
    if ( !checkVulkan( vkEnumeratePhysicalDevices( instance, &count, physicalDevices.data() ), "physical-device listing" ) ) {
        vkDestroyInstance( instance, nullptr );
        return devices;
    }

    for ( uint32_t index = 0; index < count; ++index ) {
        VkPhysicalDeviceProperties properties{};
        vkGetPhysicalDeviceProperties( physicalDevices[index], &properties );
        if ( properties.vendorID != AMD_VENDOR_ID || findComputeQueueFamily( physicalDevices[index] ) < 0 ) {
            continue;
        }

        devices.push_back( { index, properties.deviceName } );
    }

    vkDestroyInstance( instance, nullptr );
    return devices;
}

AmdVulkanContext* initAmdVulkan( uint32_t physicalDeviceIndex ) {
    AmdVulkanContext* context = new AmdVulkanContext();
    context->instance = createInstance();
    if ( context->instance == VK_NULL_HANDLE ) {
        cleanupAmdVulkan( context );
        return nullptr;
    }

    uint32_t count = 0;
    if ( !checkVulkan( vkEnumeratePhysicalDevices( context->instance, &count, nullptr ), "physical-device enumeration" ) ||
         physicalDeviceIndex >= count ) {
        cleanupAmdVulkan( context );
        return nullptr;
    }

    std::vector<VkPhysicalDevice> devices( count );
    if ( !checkVulkan( vkEnumeratePhysicalDevices( context->instance, &count, devices.data() ), "physical-device listing" ) ) {
        cleanupAmdVulkan( context );
        return nullptr;
    }

    context->physicalDevice = devices[physicalDeviceIndex];
    VkPhysicalDeviceProperties properties{};
    vkGetPhysicalDeviceProperties( context->physicalDevice, &properties );
    int queueFamily = findComputeQueueFamily( context->physicalDevice );
    if ( properties.vendorID != AMD_VENDOR_ID || queueFamily < 0 ) {
        std::cerr << "Vulkan AMD backend could not find the selected AMD compute device" << std::endl;
        cleanupAmdVulkan( context );
        return nullptr;
    }

    context->queueFamilyIndex = static_cast<uint32_t>( queueFamily );
    float priority = 1.0f;
    VkDeviceQueueCreateInfo queueInfo{};
    queueInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queueInfo.queueFamilyIndex = context->queueFamilyIndex;
    queueInfo.queueCount = 1;
    queueInfo.pQueuePriorities = &priority;

    VkDeviceCreateInfo deviceInfo{};
    deviceInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    deviceInfo.queueCreateInfoCount = 1;
    deviceInfo.pQueueCreateInfos = &queueInfo;
    if ( !checkVulkan( vkCreateDevice( context->physicalDevice, &deviceInfo, nullptr, &context->device ), "logical-device creation" ) ) {
        cleanupAmdVulkan( context );
        return nullptr;
    }
    vkGetDeviceQueue( context->device, context->queueFamilyIndex, 0, &context->queue );

    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = WORK_ITEM_COUNT * sizeof( float );
    bufferInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if ( !checkVulkan( vkCreateBuffer( context->device, &bufferInfo, nullptr, &context->outputBuffer ), "storage-buffer creation" ) ) {
        cleanupAmdVulkan( context );
        return nullptr;
    }

    VkMemoryRequirements memoryRequirements{};
    vkGetBufferMemoryRequirements( context->device, context->outputBuffer, &memoryRequirements );
    uint32_t memoryType = findMemoryType( context->physicalDevice, memoryRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT );
    if ( memoryType == UINT32_MAX ) {
        memoryType = findMemoryType( context->physicalDevice, memoryRequirements.memoryTypeBits, 0 );
    }
    if ( memoryType == UINT32_MAX ) {
        std::cerr << "Vulkan AMD backend could not allocate storage-buffer memory" << std::endl;
        cleanupAmdVulkan( context );
        return nullptr;
    }

    VkMemoryAllocateInfo allocationInfo{};
    allocationInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocationInfo.allocationSize = memoryRequirements.size;
    allocationInfo.memoryTypeIndex = memoryType;
    if ( !checkVulkan( vkAllocateMemory( context->device, &allocationInfo, nullptr, &context->outputMemory ), "storage-memory allocation" ) ||
         !checkVulkan( vkBindBufferMemory( context->device, context->outputBuffer, context->outputMemory, 0 ), "storage-buffer binding" ) ) {
        cleanupAmdVulkan( context );
        return nullptr;
    }

    VkDescriptorSetLayoutBinding binding{};
    binding.binding = 0;
    binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    binding.descriptorCount = 1;
    binding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = 1;
    layoutInfo.pBindings = &binding;
    if ( !checkVulkan( vkCreateDescriptorSetLayout( context->device, &layoutInfo, nullptr, &context->descriptorSetLayout ), "descriptor-set layout creation" ) ) {
        cleanupAmdVulkan( context );
        return nullptr;
    }

    VkDescriptorPoolSize poolSize{};
    poolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    poolSize.descriptorCount = 1;
    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.maxSets = 1;
    poolInfo.poolSizeCount = 1;
    poolInfo.pPoolSizes = &poolSize;
    if ( !checkVulkan( vkCreateDescriptorPool( context->device, &poolInfo, nullptr, &context->descriptorPool ), "descriptor-pool creation" ) ) {
        cleanupAmdVulkan( context );
        return nullptr;
    }

    VkDescriptorSetAllocateInfo descriptorInfo{};
    descriptorInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    descriptorInfo.descriptorPool = context->descriptorPool;
    descriptorInfo.descriptorSetCount = 1;
    descriptorInfo.pSetLayouts = &context->descriptorSetLayout;
    if ( !checkVulkan( vkAllocateDescriptorSets( context->device, &descriptorInfo, &context->descriptorSet ), "descriptor-set allocation" ) ) {
        cleanupAmdVulkan( context );
        return nullptr;
    }

    VkDescriptorBufferInfo bufferDescriptor{};
    bufferDescriptor.buffer = context->outputBuffer;
    bufferDescriptor.range = VK_WHOLE_SIZE;
    VkWriteDescriptorSet descriptorWrite{};
    descriptorWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptorWrite.dstSet = context->descriptorSet;
    descriptorWrite.dstBinding = 0;
    descriptorWrite.descriptorCount = 1;
    descriptorWrite.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptorWrite.pBufferInfo = &bufferDescriptor;
    vkUpdateDescriptorSets( context->device, 1, &descriptorWrite, 0, nullptr );

    context->shader = loadShaderModule( context->device );
    if ( context->shader == VK_NULL_HANDLE ) {
        cleanupAmdVulkan( context );
        return nullptr;
    }

    VkPushConstantRange pushRange{};
    pushRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pushRange.size = sizeof( uint32_t );
    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &context->descriptorSetLayout;
    pipelineLayoutInfo.pushConstantRangeCount = 1;
    pipelineLayoutInfo.pPushConstantRanges = &pushRange;
    if ( !checkVulkan( vkCreatePipelineLayout( context->device, &pipelineLayoutInfo, nullptr, &context->pipelineLayout ), "pipeline-layout creation" ) ) {
        cleanupAmdVulkan( context );
        return nullptr;
    }

    VkPipelineShaderStageCreateInfo stageInfo{};
    stageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stageInfo.module = context->shader;
    stageInfo.pName = "main";
    VkComputePipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineInfo.stage = stageInfo;
    pipelineInfo.layout = context->pipelineLayout;
    if ( !checkVulkan( vkCreateComputePipelines( context->device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &context->pipeline ), "compute-pipeline creation" ) ) {
        cleanupAmdVulkan( context );
        return nullptr;
    }

    VkCommandPoolCreateInfo commandPoolInfo{};
    commandPoolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    commandPoolInfo.queueFamilyIndex = context->queueFamilyIndex;
    commandPoolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    if ( !checkVulkan( vkCreateCommandPool( context->device, &commandPoolInfo, nullptr, &context->commandPool ), "command-pool creation" ) ) {
        cleanupAmdVulkan( context );
        return nullptr;
    }

    VkCommandBufferAllocateInfo commandBufferInfo{};
    commandBufferInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    commandBufferInfo.commandPool = context->commandPool;
    commandBufferInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    commandBufferInfo.commandBufferCount = 1;
    if ( !checkVulkan( vkAllocateCommandBuffers( context->device, &commandBufferInfo, &context->commandBuffer ), "command-buffer allocation" ) ) {
        cleanupAmdVulkan( context );
        return nullptr;
    }

    return context;
}

void cleanupAmdVulkan( AmdVulkanContext* context ) {
    if ( !context ) {
        return;
    }

    if ( context->device != VK_NULL_HANDLE ) {
        vkDeviceWaitIdle( context->device );
        if ( context->commandPool != VK_NULL_HANDLE ) vkDestroyCommandPool( context->device, context->commandPool, nullptr );
        if ( context->pipeline != VK_NULL_HANDLE ) vkDestroyPipeline( context->device, context->pipeline, nullptr );
        if ( context->pipelineLayout != VK_NULL_HANDLE ) vkDestroyPipelineLayout( context->device, context->pipelineLayout, nullptr );
        if ( context->shader != VK_NULL_HANDLE ) vkDestroyShaderModule( context->device, context->shader, nullptr );
        if ( context->descriptorPool != VK_NULL_HANDLE ) vkDestroyDescriptorPool( context->device, context->descriptorPool, nullptr );
        if ( context->descriptorSetLayout != VK_NULL_HANDLE ) vkDestroyDescriptorSetLayout( context->device, context->descriptorSetLayout, nullptr );
        if ( context->outputBuffer != VK_NULL_HANDLE ) vkDestroyBuffer( context->device, context->outputBuffer, nullptr );
        if ( context->outputMemory != VK_NULL_HANDLE ) vkFreeMemory( context->device, context->outputMemory, nullptr );
        vkDestroyDevice( context->device, nullptr );
    }
    if ( context->instance != VK_NULL_HANDLE ) {
        vkDestroyInstance( context->instance, nullptr );
    }
    delete context;
}

bool launchAmdVulkanKernel( AmdVulkanContext* context, int gridSize, int blockSize, int iterations ) {
    if ( !context || blockSize != 256 || gridSize <= 0 || static_cast<uint32_t>( gridSize ) * blockSize > WORK_ITEM_COUNT ) {
        return false;
    }

    if ( !checkVulkan( vkResetCommandBuffer( context->commandBuffer, 0 ), "command-buffer reset" ) ) {
        return false;
    }

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    if ( !checkVulkan( vkBeginCommandBuffer( context->commandBuffer, &beginInfo ), "command-buffer begin" ) ) {
        return false;
    }

    vkCmdBindPipeline( context->commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, context->pipeline );
    vkCmdBindDescriptorSets( context->commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, context->pipelineLayout,
                             0, 1, &context->descriptorSet, 0, nullptr );
    uint32_t iterationCount = static_cast<uint32_t>( iterations );
    vkCmdPushConstants( context->commandBuffer, context->pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT,
                        0, sizeof( iterationCount ), &iterationCount );
    vkCmdDispatch( context->commandBuffer, static_cast<uint32_t>( gridSize ), 1, 1 );

    if ( !checkVulkan( vkEndCommandBuffer( context->commandBuffer ), "command-buffer end" ) ) {
        return false;
    }

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &context->commandBuffer;
    return checkVulkan( vkQueueSubmit( context->queue, 1, &submitInfo, VK_NULL_HANDLE ), "compute submission" ) &&
           checkVulkan( vkQueueWaitIdle( context->queue ), "compute completion" );
}
