#ifndef AMD_VULKAN_WORKER_H
#define AMD_VULKAN_WORKER_H

#include <cstdint>
#include <string>
#include <vector>

struct AmdVulkanContext;

struct AmdVulkanDevice {
    uint32_t physicalDeviceIndex;
    std::string name;
};

std::vector<AmdVulkanDevice> enumerateAmdVulkanDevices();
AmdVulkanContext* initAmdVulkan( uint32_t physicalDeviceIndex );
void cleanupAmdVulkan( AmdVulkanContext* context );
bool launchAmdVulkanKernel( AmdVulkanContext* context, int gridSize, int blockSize, int iterations );

#endif
