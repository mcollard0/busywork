// Workaround for rsqrt conflict between CUDA 13.0 and glibc 2.40+
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wignored-attributes"

// Only include essential CUDA headers
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <nvml.h>

#pragma GCC diagnostic pop

// Forward declare our struct before including the header
struct GpuContext;

// Include our header
extern "C" {
    GpuContext* initGpu();
    void cleanupGpu( GpuContext* ctx );
    void launchGpuKernel( GpuContext* ctx, int gridSize, int blockSize, int iterations );
    double queryGpuLoad();
    int queryGpuCoreCount();
}

struct GpuContext {
    float* deviceData;
    nvmlDevice_t nvmlDevice;
    bool nvmlInitialized;
};

// CUDA kernel for heat generation
__global__ void heatKernel( int iterations, float* output ) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float value = 1.0f + idx * 0.0001f;
    
    // Intensive floating point computations
    for ( int i = 0; i < iterations; ++i ) {
        value = sinf( value ) * cosf( value );
        value = sqrtf( fabsf( value ) + 1.0f );
        value = fmaf( value, value, 1.0f );
    };
    
    // Prevent optimization
    if ( idx == 0 ) {
        *output = value;
    };
};

GpuContext* initGpu() {
    GpuContext* ctx = new GpuContext();
    ctx->nvmlInitialized = false;
    
    // Allocate device memory
    cudaError_t err = cudaMalloc( &ctx->deviceData, sizeof( float ) );
    if ( err != cudaSuccess ) {
        delete ctx;
        return nullptr;
    };
    
    // Initialize NVML for monitoring
    if ( nvmlInit() == NVML_SUCCESS ) {
        if ( nvmlDeviceGetHandleByIndex( 0, &ctx->nvmlDevice ) == NVML_SUCCESS ) {
            ctx->nvmlInitialized = true;
        };
    };
    
    return ctx;
};

void cleanupGpu( GpuContext* ctx ) {
    if ( !ctx ) return;
    
    if ( ctx->deviceData ) {
        cudaFree( ctx->deviceData );
    };
    
    if ( ctx->nvmlInitialized ) {
        nvmlShutdown();
    };
    
    delete ctx;
};

void launchGpuKernel( GpuContext* ctx, int gridSize, int blockSize, int iterations ) {
    if ( !ctx || !ctx->deviceData ) return;
    
    heatKernel<<<gridSize, blockSize>>>( iterations, ctx->deviceData );
    cudaDeviceSynchronize();
};

double queryGpuLoad() {
    static nvmlDevice_t device;
    static bool initialized = false;
    static int smCount = 0;
    
    if ( !initialized ) {
        if ( nvmlInit() == NVML_SUCCESS ) {
            if ( nvmlDeviceGetHandleByIndex( 0, &device ) == NVML_SUCCESS ) {
                initialized = true;
                // Get SM count
                unsigned int count = 0;
                if ( nvmlDeviceGetNumGpuCores( device, &count ) == NVML_SUCCESS ) {
                    smCount = static_cast<int>( count );
                } else {
                    // Fallback: try to get SM count via different method
                    int devId;
                    cudaGetDevice( &devId );
                    cudaDeviceProp prop;
                    cudaGetDeviceProperties( &prop, devId );
                    smCount = prop.multiProcessorCount;
                };
            };
        };
    };
    
    if ( !initialized || smCount == 0 ) {
        return 0.0;
    };
    
    nvmlUtilization_t utilization;
    if ( nvmlDeviceGetUtilizationRates( device, &utilization ) == NVML_SUCCESS ) {
        // Convert 0-100% to "SMs worth of load"
        return ( utilization.gpu / 100.0 ) * smCount;
    };
    
    return 0.0;
};

int queryGpuCoreCount() {
    nvmlDevice_t device;
    
    if ( nvmlInit() != NVML_SUCCESS ) {
        return 0;
    };
    
    if ( nvmlDeviceGetHandleByIndex( 0, &device ) != NVML_SUCCESS ) {
        nvmlShutdown();
        return 0;
    };
    
    // Try NVML first (may not support NumGpuCores on all devices)
    unsigned int coreCount = 0;
    if ( nvmlDeviceGetNumGpuCores( device, &coreCount ) == NVML_SUCCESS ) {
        nvmlShutdown();
        return static_cast<int>( coreCount );
    };
    
    // Fallback: use CUDA runtime to get multiprocessor count
    int devId;
    cudaGetDevice( &devId );
    cudaDeviceProp prop;
    if ( cudaGetDeviceProperties( &prop, devId ) == cudaSuccess ) {
        nvmlShutdown();
        return prop.multiProcessorCount;
    };
    
    nvmlShutdown();
    return 0;
};
