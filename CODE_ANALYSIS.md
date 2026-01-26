# Busywork - Code Analysis & Documentation

## Executive Summary

**Busywork** is a C++17/CUDA application designed to generate controlled computational heat through CPU and GPU workloads for office warming purposes. The project uses dynamic load management to maintain target utilization levels through real-time monitoring and adaptive thread control.

**Primary Language:** C++ (with CUDA extensions)  
**Build System:** GNU Make  
**Target Platform:** Linux (x86_64 with optional NVIDIA GPU)  
**Total Source Files:** 3 core files (main.cpp, gpu_worker.h, gpu_kernel.cu)

---

## Project Purpose & Functionality

The application serves as a "space heater" by generating computational workloads that produce heat as a byproduct. It provides:

1. **Controlled CPU Heat Generation** - Spawns worker threads performing intensive floating-point operations
2. **Controlled GPU Heat Generation** - Launches CUDA kernels for GPU-based computation (when available)
3. **Adaptive Load Management** - Monitors actual utilization and dynamically adjusts workload intensity
4. **User-Configurable Targets** - Command-line options to set desired CPU/GPU utilization levels

---

## Languages & Technologies

### Core Technologies
- **C++17** - Primary implementation language
- **CUDA C++14** - GPU kernel implementation
- **POSIX Threads (pthread)** - Multi-threading support
- **NVML (NVIDIA Management Library)** - GPU monitoring

### Build Tools
- **g++** - C++ compiler with C++17 support
- **nvcc** - NVIDIA CUDA compiler
- **GNU Make** - Build automation

### System Dependencies
- Linux `/proc/stat` for CPU monitoring
- CUDA Toolkit (optional, for GPU support)
- NVIDIA driver with NVML

---

## Architecture Overview

### Design Pattern
The application follows a **monitor-and-control** architecture with multiple concurrent worker threads:

```
Main Thread
    ├─> CPU Worker Threads (N-1, where N = core count)
    ├─> GPU Worker Threads (configurable count)
    └─> Monitor Thread (adjusts workload every 5 seconds)
```

### Key Design Decisions

1. **Separation of CPU/GPU builds** - Makefile detects CUDA and builds appropriate binary
2. **Thread-safe state management** - Uses `std::atomic<bool>` for thread activation flags
3. **Duty cycle control** - GPU load managed through work/sleep ratio rather than kernel parameter tuning
4. **Hysteresis-based adjustment** - Prevents oscillation by using threshold bands (±0.5 threads for CPU, ±0.5 units for GPU)
5. **Graceful degradation** - CPU-only functionality when GPU unavailable

### Component Interaction

```
User Input → BusyWork → [CPU Workers] → CpuMonitor → Load Feedback
                      ↓
                   [GPU Workers] → CUDA Kernels → NVML → Load Feedback
```

---

## Source Code Structure

### File Organization

```
busywork/
├── src/
│   ├── main.cpp          # Main program, BusyWork & CpuMonitor classes
│   ├── gpu_worker.h      # GPU interface header (C linkage)
│   └── gpu_kernel.cu     # CUDA kernel implementation
├── build/                # Compiled object files
├── backup/               # Source code backups
├── logs/                 # Build and runtime logs
├── antigravity/          # CUDA SDK headers (vendor includes)
├── claude/               # (purpose unclear - possibly AI assistance logs)
├── opencode/             # (purpose unclear - possibly alternative implementations)
├── Makefile              # Build configuration
├── README.md             # User documentation
├── ARCHITECTURE.md       # Original design specification
├── WARP.md               # Development notes and technical details
└── WARP-EVALUATION.md    # (evaluation notes)
```

---

## Detailed Component Analysis

### 1. CpuMonitor Class (main.cpp:15-54)

**Purpose:** Monitors system-wide CPU utilization using Linux `/proc/stat`

**Key Methods:**
- `getCpuLoad()` - Calculates CPU load as "threads worth of work"

**Algorithm:**
1. Reads CPU time counters from `/proc/stat` (user, nice, system, idle, etc.)
2. Calculates delta between current and previous measurement
3. Computes utilization ratio: `(total - idle) / total`
4. Scales to "thread units": `utilization × cpuCount`

**Return Value:** Double representing equivalent fully-busy threads (e.g., 3.5 = "three and a half threads worth of load")

**Limitations:** 
- Linux-specific (relies on `/proc/stat`)
- System-wide measurement (not per-process)

---

### 2. BusyWork Class (main.cpp:56-323)

**Purpose:** Main orchestration class managing all workers and monitoring

**Private Members:**
- `running` - Atomic flag for shutdown signaling
- `cpuGoal`, `gpuGoal` - Target utilization levels
- `cpuThreads`, `gpuThreads` - Worker thread containers
- `cpuThreadStates` - Activation flags for individual CPU workers
- `gpuDutyCycle` - GPU work intensity (0.0-1.0)
- `cpuMonitor` - CPU monitoring instance

**Key Methods:**

#### cpuWork() (line 74-90)
- **Purpose:** Worker thread function performing CPU-intensive computations
- **Parameters:** `threadId` (unused), `active` (atomic flag controlling execution)
- **Workload:** 100,000 iterations of `sin(i) × cos(i)` → `sqrt(abs(result))`
- **Behavior:** Runs continuously when `active`, sleeps when deactivated

#### monitor() (line 93-157)
- **Purpose:** Monitoring loop that adjusts workload every 5 seconds
- **CPU Adjustment:** Activates/deactivates threads based on load vs. target
- **GPU Adjustment:** Modifies duty cycle in 0.05 increments
- **Hysteresis:** ±0.5 threshold prevents oscillation

#### gpuWorker() (line 173-203)
- **Purpose:** Host thread that launches GPU kernels repeatedly
- **Parameters:** Grid size 1024, block size 256, 100k iterations
- **Duty Cycle:** Sleeps proportionally to `(1.0 - duty)` after each kernel

#### start() (line 269-296)
- **Purpose:** Initializes and launches all worker threads
- **CPU Workers:** Spawns N-1 threads (where N = core count)
- **GPU Workers:** Spawns configurable count (default: 1)

#### stop() (line 298-316)
- **Purpose:** Signals shutdown and joins all threads

---

### 3. GPU Interface (gpu_worker.h:1-32)

**Purpose:** C-linkage interface for CUDA functionality (enables C++ ↔ CUDA interop)

**Functions Declared:**
- `initGpu()` - Creates GPU context (allocates memory, initializes NVML)
- `cleanupGpu()` - Releases GPU resources
- `launchGpuKernel()` - Executes heat generation kernel
- `queryGpuLoad()` - Returns current GPU utilization
- `queryGpuCoreCount()` - Returns streaming multiprocessor count

**Design Note:** Uses opaque `GpuContext*` pointer to hide CUDA types from C++ header

---

### 4. GPU Kernel Implementation (gpu_kernel.cu:1-159)

#### GpuContext Structure (line 24-28)
- `deviceData` - Device memory pointer for output
- `nvmlDevice` - NVML device handle
- `nvmlInitialized` - Initialization status flag

#### heatKernel() (line 31-46)
**Purpose:** CUDA kernel performing intensive floating-point math  
**Workload per thread:**
```cuda
for (i in 0..iterations) {
    value = sin(value) × cos(value)
    value = sqrt(|value| + 1.0)
    value = fma(value, value, 1.0)  // value² + 1.0
}
```
**Thread count per launch:** 1024 × 256 = 262,144 threads

#### initGpu() (line 48-67)
1. Allocates device memory for single float (output sink)
2. Initializes NVML for monitoring
3. Gets handle for GPU device 0
4. Returns context or `nullptr` on failure

#### queryGpuLoad() (line 90-126)
- Uses static variables to cache device handle and SM count
- Queries `nvmlDeviceGetUtilizationRates()` for percentage
- Converts to "SMs worth of load": `(utilization% / 100) × smCount`
- Falls back to `cudaDeviceProp` if NVML SM query fails

#### queryGpuCoreCount() (line 128-158)
- Attempts `nvmlDeviceGetNumGpuCores()` first
- Falls back to `cudaDeviceProp.multiProcessorCount`
- Returns SM count for load scaling

---

## Build System (Makefile)

### Compilation Flags
- **C++:** `-std=c++17 -pthread -O3 -Wall -Wextra`
- **CUDA:** `-O3 -std=c++14 -ccbin /usr/bin/gcc-12`

### Build Process
1. Detects CUDA installation at `/usr/local/cuda`
2. If CUDA available:
   - Links: `-lcudart -lnvidia-ml`
   - Compiles `gpu_kernel.cu` → `build/gpu_kernel.o`
   - Defines `ENABLE_CUDA` preprocessor flag
3. Always compiles `main.cpp` → `build/main.o`
4. Links objects into single `busywork` binary

### Directory Creation
- Automatically creates `build/`, `backup/src/`, `logs/` on build

---

## Suggestions & Improvements

### Functionality Enhancements

1. **Cross-Platform Support**
   - Abstract CPU monitoring into platform-specific modules
   - Add Windows Performance Counters support
   - Add macOS `host_statistics()` support

2. **Multi-GPU Support**
   - Extend to enumerate and use all available GPUs
   - Add per-GPU worker allocation
   - Implement load balancing across devices

3. **Temperature-Based Control**
   - Query CPU/GPU temperatures via sensors
   - Implement PID controller for target temperature
   - Add thermal safety limits with automatic shutdown

4. **Power Monitoring**
   - Integrate NVML power queries
   - Report watts consumed (useful for efficiency analysis)
   - Add power budget constraints

5. **Configuration File**
   - YAML/JSON config for persistent settings
   - Profile support (e.g., "office-hours", "weekend")
   - Scheduled operation (time-based activation)

6. **Web Interface**
   - REST API for remote control
   - Real-time monitoring dashboard
   - Prometheus/Grafana integration

7. **Adaptive Kernel Tuning**
   - Auto-detect optimal grid/block sizes
   - Adjust iteration count based on GPU architecture
   - Per-device kernel configurations

8. **Improved Load Metrics**
   - Per-core CPU utilization display
   - GPU memory bandwidth utilization
   - Power efficiency metrics (heat/watt)

### Testing Recommendations

1. **Unit Tests**
   - Mock `/proc/stat` for CpuMonitor tests
   - Test thread activation/deactivation logic
   - Validate duty cycle calculations

2. **Integration Tests**
   - Test CPU load convergence to target
   - Test GPU load convergence to target
   - Test graceful degradation (no GPU)
   - Test multi-GPU worker scenarios

3. **Stress Tests**
   - Rapid target changes
   - Maximum thread counts
   - Long-duration stability (24+ hours)

4. **Platform Tests**
   - Various Linux distributions
   - Different CUDA versions
   - AMD GPUs (via ROCm) - requires porting

### Code Quality Improvements

1. **Error Handling**
   - Add exception handling for resource allocation
   - Validate command-line arguments (range checks)
   - Handle NVML errors more gracefully

2. **Logging System**
   - Structured logging (timestamps, levels)
   - Configurable verbosity
   - Rotating log files

3. **Documentation**
   - Add Doxygen comments to functions
   - Generate API documentation
   - Add architecture diagrams

4. **Code Organization**
   - Extract CpuMonitor to separate file
   - Create abstract WorkerInterface
   - Separate monitoring logic from BusyWork class

5. **Performance Optimization**
   - Profile actual heat generation efficiency
   - Experiment with different math operations
   - GPU kernel occupancy analysis

6. **Safety Features**
   - Watchdog timer for runaway threads
   - Thermal throttling integration
   - Resource limit enforcement (cgroups)

---

## Enhancement Ideas

### Advanced Features

1. **Machine Learning Integration**
   - Use CUDA tensor cores for ML workloads
   - Train dummy models as heat source
   - Benchmark different ML operations for heat efficiency

2. **Distributed Heating**
   - Network protocol for coordinated multi-machine heating
   - Cluster-wide target temperature management
   - Failover and redundancy

3. **Smart Home Integration**
   - MQTT/HomeAssistant integration
   - Temperature sensor integration
   - Automated scheduling based on occupancy

4. **Cryptocurrency Mining Mode**
   - Useful computation instead of synthetic heat
   - Mine low-difficulty coins during heating season
   - Offset energy costs

5. **Acoustic Optimization**
   - Adjust workload to minimize fan noise
   - Target optimal temp/noise ratio
   - Fan curve optimization

6. **Energy Economics**
   - Cost calculator (electricity vs. traditional heating)
   - Carbon footprint comparison
   - ROI analysis for hardware investment

---

## Function/Method Reference Table

| Function/Method | File | Line | Description |
|-----------------|------|------|-------------|
| `CpuMonitor::CpuMonitor()` | main.cpp | 22 | Constructor initializing CPU count |
| `CpuMonitor::getCpuLoad()` | main.cpp | 25 | Reads `/proc/stat` and calculates CPU utilization in thread-equivalent units |
| `BusyWork::BusyWork()` | main.cpp | 211 | Constructor initializing CPU/GPU goals, detecting hardware capabilities |
| `BusyWork::cpuWork()` | main.cpp | 74 | CPU worker thread performing intensive floating-point calculations (sin/cos/sqrt) |
| `BusyWork::monitor()` | main.cpp | 93 | Monitoring loop running every 5s, adjusts CPU thread activation and GPU duty cycle |
| `BusyWork::printStatus()` | main.cpp | 159 | Outputs current CPU/GPU load, goals, and active worker counts to console |
| `BusyWork::gpuWorker()` | main.cpp | 173 | GPU worker thread launching CUDA kernels with duty-cycle controlled sleep |
| `BusyWork::getGpuLoad()` | main.cpp | 205 | Wrapper calling `queryGpuLoad()` from CUDA module |
| `BusyWork::printCapabilities()` | main.cpp | 247 | Displays system hardware capabilities (CPU threads, GPU SMs) and current configuration |
| `BusyWork::start()` | main.cpp | 269 | Spawns all CPU/GPU worker threads and monitor thread, prints initial status |
| `BusyWork::stop()` | main.cpp | 298 | Sets running flag to false and joins all threads for graceful shutdown |
| `BusyWork::waitForSignal()` | main.cpp | 318 | Blocks on stdin waiting for Enter key to trigger shutdown |
| `printUsage()` | main.cpp | 325 | Displays command-line help message with available options |
| `main()` | main.cpp | 336 | Entry point: parses arguments, creates BusyWork instance, starts and waits |
| `initGpu()` | gpu_kernel.cu | 48 | Allocates GPU memory, initializes NVML, returns GpuContext pointer |
| `cleanupGpu()` | gpu_kernel.cu | 69 | Frees GPU memory, shuts down NVML, deletes context |
| `heatKernel()` | gpu_kernel.cu | 31 | CUDA kernel executing intensive FP operations (sin/cos/sqrt/fma) per thread |
| `launchGpuKernel()` | gpu_kernel.cu | 83 | Launches heatKernel with specified grid/block size and synchronizes |
| `queryGpuLoad()` | gpu_kernel.cu | 90 | Queries GPU utilization via NVML, converts to "SMs worth" metric |
| `queryGpuCoreCount()` | gpu_kernel.cu | 128 | Returns GPU streaming multiprocessor count via NVML or CUDA runtime |

---

## Testing Checklist

### Build Tests
- [ ] CPU-only build (without CUDA)
- [ ] CUDA-enabled build
- [ ] Build on various GCC versions
- [ ] Build with different CUDA toolkit versions

### Functional Tests
- [ ] Default settings execution
- [ ] Custom CPU goal (0.0, 0.5, 0.8, 1.0)
- [ ] Custom GPU goal (0.0, 0.5, 0.8, 1.0)
- [ ] Multiple GPU worker threads (1, 2, 4, 8)
- [ ] Graceful shutdown (Enter key)
- [ ] Invalid arguments handling

### Performance Tests
- [ ] CPU load convergence time
- [ ] CPU load accuracy (±5% of target)
- [ ] GPU load convergence time
- [ ] GPU load accuracy (±5% of target)
- [ ] Stability over 1 hour
- [ ] Stability over 24 hours

### Edge Cases
- [ ] Single-core system
- [ ] No GPU present
- [ ] GPU with very few SMs
- [ ] Rapid goal changes
- [ ] Goal values at boundaries (0.0, 1.0)

---

## Conclusion

Busywork is a well-architected thermal management utility demonstrating solid C++/CUDA integration practices. The codebase is clean, modular, and follows good separation of concerns. The adaptive load control algorithm is simple yet effective, using hysteresis to prevent oscillation.

**Strengths:**
- Clean architecture with clear responsibilities
- Robust thread management
- Good fallback behavior (CPU-only mode)
- Efficient build system with CUDA detection
- Excellent inline documentation

**Areas for Improvement:**
- Platform portability (Linux-only)
- Single GPU limitation
- Hardcoded kernel parameters
- Limited error handling
- No persistent configuration

**Overall Assessment:** Production-quality for intended use case, with clear paths for extension to more advanced scenarios.
