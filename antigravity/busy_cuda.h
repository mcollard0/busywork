#ifndef BUSY_CUDA_H
#define BUSY_CUDA_H

#include <cstddef>

extern "C" {
float *gpu_alloc(size_t size);
void gpu_free(float *ptr);
void launch_gpu_kernel_async(int blocks, int threads, int iterations,
                             float *d_data);
void gpu_sync();
void get_device_count(int *count);
void set_device(int device);
}

#endif
