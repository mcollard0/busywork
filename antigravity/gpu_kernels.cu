#define __MATH_FUNCTIONS_H__
#include "busy_cuda.h"
#include <cuda_runtime.h>

__global__ void busy_kernel(int iterations, float *dummy_out) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  float val = 1.0f + idx * 0.0001f;

  // Heavy ALU loop
  for (int i = 0; i < iterations; ++i) {
    val = sinf(val) * cosf(val);
    val = sqrtf(fabsf(val) + 1.0f);
    val = fmaf(val, val, 1.0f);
  }

  if (idx == 0)
    *dummy_out = val;
}

extern "C" {

float *gpu_alloc(size_t size) {
  float *ptr;
  cudaMalloc(&ptr, size);
  return ptr;
}

void gpu_free(float *ptr) { cudaFree(ptr); }

void launch_gpu_kernel_async(int blocks, int threads, int iterations,
                             float *d_data) {
  busy_kernel<<<blocks, threads>>>(iterations, d_data);
}

void gpu_sync() { cudaDeviceSynchronize(); }

void get_device_count(int *count) { cudaGetDeviceCount(count); }

void set_device(int device) { cudaSetDevice(device); }
}
