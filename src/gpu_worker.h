#ifndef GPU_WORKER_H
#define GPU_WORKER_H

#include <cstddef>

struct GpuContext;

#ifdef __cplusplus
extern "C" {
#endif

// Initialize GPU context for worker thread
GpuContext* initGpu();

// Cleanup GPU resources
void cleanupGpu( GpuContext* ctx );

// Launch compute kernel on GPU
void launchGpuKernel( GpuContext* ctx, int gridSize, int blockSize, int iterations );

// Query current GPU load (as number of SMs worth)
double queryGpuLoad();

// Query number of GPU multiprocessors (SMs)
int queryGpuCoreCount();

#ifdef __cplusplus
}
#endif

#endif
