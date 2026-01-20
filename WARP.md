# Busywork - Warp Development Documentation

## Overview
Busywork is a C++ application that generates controlled CPU and GPU computational workloads to produce heat for office warming.

## Implementation Status

### ✅ Completed Features

#### CPU Heat Generation
- Multi-threaded worker pool with dynamic activation/deactivation
- Real-time CPU monitoring via `/proc/stat`
- Automatic load adjustment every 5 seconds
- Intensive floating-point operations (sin, cos, sqrt)
- Thread-safe state management with atomic operations
- Configurable target utilization (0.0-1.0)

#### GPU Heat Generation (CUDA)
- CUDA kernel implementation with configurable grid/block sizes
- NVML integration for GPU load monitoring
- Duty cycle-based load control
- Multi-threaded GPU workers
- Automatic fallback if GPU unavailable

#### Build System
- Makefile with automatic CUDA detection
- Separate CPU-only and CUDA builds
- Automatic directory structure creation
- Clean separation of concerns

#### Command Line Interface
- `--cpugoal` for CPU utilization target
- `--gpugoal` for GPU utilization target
- `--gputhreads` for GPU worker count
- Help system

### 📋 Architecture

#### Key Classes

**CpuMonitor**
- Reads `/proc/stat` for system-wide CPU usage
- Tracks delta between measurements
- Returns load as fraction (0.0-1.0)

**BusyWork**
- Main orchestration class
- Manages CPU and GPU worker threads
- Runs monitoring loop
- Handles graceful shutdown

#### Threading Model
- Main thread: User input handling
- Monitor thread: Load monitoring and adjustment
- CPU worker threads: Heat generation (N-1 cores)
- GPU worker threads: CUDA kernel launches (configurable)

#### GPU Implementation
- `GpuContext`: Per-thread GPU state (device memory, NVML handle)
- `heatKernel`: CUDA kernel performing intensive math
- Duty cycle control: Adjusts work/sleep ratio based on actual vs target load

### 🔧 Technical Details

#### CPU Load Algorithm
```
1. Read /proc/stat CPU counters
2. Calculate utilization = (total_active) / (total_time)
3. If load < target - 0.02: Activate idle thread
4. If load > target + 0.02: Deactivate active thread
5. Repeat every 5 seconds
```

#### GPU Load Algorithm
```
1. Launch CUDA kernel (intensive FP ops)
2. Sync GPU
3. Query GPU utilization via NVML
4. If load < target - 0.05: Increase duty cycle by 0.05
5. If load > target + 0.05: Decrease duty cycle by 0.05
6. Sleep proportional to (1 - duty_cycle)
7. Repeat
```

#### Kernel Configuration
- Grid size: 1024 blocks
- Block size: 256 threads
- Total threads: 262,144 per launch
- Iterations: 100,000 FP operations per thread

### 📁 File Organization

```
src/main.cpp          - Main program, CPU workers, monitoring
src/gpu_worker.h      - GPU interface definitions
src/gpu_kernel.cu     - CUDA kernel and wrappers
build/                - Compiled objects
backup/src/           - Source backups
logs/                 - Build logs
```

### 🎯 Usage Patterns

**Basic CPU heating:**
```bash
./busywork
# Uses (N-1) cores at maximum capacity
```

**Controlled CPU load:**
```bash
./busywork --cpugoal 0.6
# Target 60% system-wide CPU utilization
```

**CPU + GPU heating:**
```bash
./busywork_cuda --cpugoal 0.8 --gpugoal 0.9
# 80% CPU, 90% GPU
```

**Multi-GPU worker:**
```bash
./busywork_cuda --gputhreads 4 --gpugoal 0.95
# 4 host threads launching GPU work
```

### 🚧 Known Limitations

1. **Linux-only**: CPU monitoring uses `/proc/stat`
2. **Single GPU**: NVML queries GPU 0 only
3. **No persistence**: Settings not saved between runs
4. **Fixed kernel params**: Grid/block sizes hardcoded
5. **Coarse adjustment**: Load changes in discrete thread increments

### 🔮 Potential Enhancements

1. Cross-platform CPU monitoring (Windows, macOS)
2. Multi-GPU support with load balancing
3. Configuration file support
4. Web dashboard for monitoring
5. Adaptive kernel sizing based on GPU capabilities
6. Temperature-based feedback control
7. Power consumption monitoring
8. Scheduled operation (time-based on/off)

### 🧪 Testing

Build and test CPU version:
```bash
make clean && make
./busywork --cpugoal 0.5 &
# Monitor with: htop, top, or mpstat
```

Build and test CUDA version (requires CUDA toolkit):
```bash
make clean && make
./busywork_cuda --cpugoal 0.5 --gpugoal 0.8 &
# Monitor with: nvidia-smi, nvtop
```

### 📝 Notes

- CPU goal is system-wide utilization (0.0 to 1.0)
- Default CPU goal: (N-1)/N where N = core count
- GPU goal is device utilization (0.0 to 1.0)
- Default GPU goal: 0.99 (maximum)
- Monitor loop runs every 5 seconds
- Hysteresis: ±2% for CPU, ±5% for GPU
