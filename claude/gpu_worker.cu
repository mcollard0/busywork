// Minimal CUDA implementation to avoid glibc rsqrt conflict
// Avoids including any standard C++ headers in CUDA code

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

// Forward declare GpuInfo since we can't include our header safely
extern "C" {

__global__ void busyKernel(float* data, int n, int iterations) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = data[idx];
        for (int i = 0; i < iterations; ++i) {
            val = __sinf(val) * __cosf(val) + __tanf(val * 0.001f);
            val = fmodf(val, 1000000.0f);
        }
        data[idx] = val;
    }
}

struct GpuContextInternal {
    float* d_data;
    int numElements;
    int blockSize;
    int numBlocks;
    cudaStream_t stream;
    int initialized;
};

static GpuContextInternal ctx = {nullptr, 0, 256, 0, nullptr, 0};

int gpuInit(int deviceId) {
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess || deviceCount == 0) {
        fprintf(stderr, "No CUDA devices found\n");
        return 0;
    }

    if (deviceId >= deviceCount) {
        fprintf(stderr, "Invalid device ID %d (have %d devices)\n", deviceId, deviceCount);
        return 0;
    }

    err = cudaSetDevice(deviceId);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to set CUDA device: %s\n", cudaGetErrorString(err));
        return 0;
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, deviceId);
    printf("GPU: %s (%d SMs)\n", prop.name, prop.multiProcessorCount);

    ctx.numElements = prop.multiProcessorCount * 1024 * 256;
    ctx.blockSize = 256;
    ctx.numBlocks = (ctx.numElements + ctx.blockSize - 1) / ctx.blockSize;

    err = cudaMalloc(&ctx.d_data, ctx.numElements * sizeof(float));
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate GPU memory: %s\n", cudaGetErrorString(err));
        return 0;
    }

    float* h_data = (float*)malloc(ctx.numElements * sizeof(float));
    for (int i = 0; i < ctx.numElements; ++i) {
        h_data[i] = (float)(i % 1000);
    }
    cudaMemcpy(ctx.d_data, h_data, ctx.numElements * sizeof(float), cudaMemcpyHostToDevice);
    free(h_data);

    cudaStreamCreate(&ctx.stream);
    ctx.initialized = 1;

    return 1;
}

void gpuCleanup() {
    if (ctx.initialized) {
        cudaStreamDestroy(ctx.stream);
        cudaFree(ctx.d_data);
        ctx.initialized = 0;
    }
}

void gpuLaunchWork(int intensity) {
    if (!ctx.initialized) return;

    int iterations = intensity * 1000;
    busyKernel<<<ctx.numBlocks, ctx.blockSize, 0, ctx.stream>>>(
        ctx.d_data, ctx.numElements, iterations
    );
}

void gpuSync() {
    if (ctx.initialized) {
        cudaStreamSynchronize(ctx.stream);
    }
}

int gpuAvailable() {
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    return (err == cudaSuccess && deviceCount > 0);
}

void gpuGetInfo(int* deviceCount, char* name, int nameLen, int* multiProcessors) {
    *deviceCount = 0;
    *multiProcessors = 0;
    name[0] = '\0';

    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
        return;
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    *deviceCount = count;
    strncpy(name, prop.name, nameLen - 1);
    name[nameLen - 1] = '\0';
    *multiProcessors = prop.multiProcessorCount;
}

} // extern "C"
