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
    double queryGpuPowerWatts();
    double queryGpuTemperature();
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

static bool getNvmlDevice( nvmlDevice_t* dev, int* smCountOut ) {
    static nvmlDevice_t device;
    static bool initialized = false;
    static int smCount = 0;
    
    if ( !initialized ) {
        if ( nvmlInit() == NVML_SUCCESS ) {
            if ( nvmlDeviceGetHandleByIndex( 0, &device ) == NVML_SUCCESS ) {
                initialized = true;
                unsigned int count = 0;
                if ( nvmlDeviceGetNumGpuCores( device, &count ) == NVML_SUCCESS ) {
                    smCount = static_cast<int>( count );
                } else {
                    int devId;
                    cudaGetDevice( &devId );
                    cudaDeviceProp prop;
                    cudaGetDeviceProperties( &prop, devId );
                    smCount = prop.multiProcessorCount;
                };
            };
        };
    };
    
    if ( !initialized ) return false;
    if ( dev ) *dev = device;
    if ( smCountOut ) *smCountOut = smCount;
    return true;
}

double queryGpuLoad() {
    nvmlDevice_t device;
    int smCount = 0;
    if ( !getNvmlDevice( &device, &smCount ) || smCount == 0 ) {
        return 0.0;
    };
    
    nvmlUtilization_t utilization;
    if ( nvmlDeviceGetUtilizationRates( device, &utilization ) == NVML_SUCCESS ) {
        // Convert 0-100% to "SMs worth of load"
        return ( utilization.gpu / 100.0 ) * smCount;
    };
    
    return 0.0;
};

double queryGpuPowerWatts() {
    nvmlDevice_t device;
    if ( !getNvmlDevice( &device, nullptr ) ) {
        return 0.0;
    };
    
    unsigned int powerMilliWatts = 0;
    if ( nvmlDeviceGetPowerUsage( device, &powerMilliWatts ) == NVML_SUCCESS ) {
        return static_cast<double>( powerMilliWatts ) / 1000.0;
    };
    
    return 0.0;
};

double queryGpuTemperature() {
    nvmlDevice_t device;
    if ( !getNvmlDevice( &device, nullptr ) ) {
        return 0.0;
    };
    
    unsigned int tempC = 0;
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
    if ( nvmlDeviceGetTemperature( device, NVML_TEMPERATURE_GPU, &tempC ) == NVML_SUCCESS ) {
        return static_cast<double>( tempC );
    };
#pragma GCC diagnostic pop
    
    return 0.0;
};

int queryGpuCoreCount() {
    nvmlDevice_t device;
    int smCount = 0;
    if ( getNvmlDevice( &device, &smCount ) ) {
        return smCount;
    };
    
    // Fallback: use CUDA runtime to get multiprocessor count
    int devId;
    if ( cudaGetDevice( &devId ) == cudaSuccess ) {
        cudaDeviceProp prop;
        if ( cudaGetDeviceProperties( &prop, devId ) == cudaSuccess ) {
            return prop.multiProcessorCount;
        };
    };
    
    return 0;
};
