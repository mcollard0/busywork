#ifndef GPU_WORKER_H
#define GPU_WORKER_H

#include <string>

struct GpuInfo {
    int deviceCount;
    std::string name;
    int multiProcessors;
};

bool gpuInit(int deviceId = 0);
void gpuCleanup();
void gpuLaunchWork(int intensity);
void gpuSync();
bool gpuAvailable();
GpuInfo gpuGetInfo();

#endif
