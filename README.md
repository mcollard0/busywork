# Busywork - CPU/GPU Heat Generator

A C++ program designed to generate computational heat for office warming via CPU and GPU utilization.

## Purpose

Generate heat through controlled CPU and GPU workloads to warm your office space efficiently.

## Building

```bash
make
```

This creates:
- `busywork` - CPU-only version (always built)
- `busywork_cuda` - CUDA-enabled version (requires CUDA toolkit)

## Usage

```bash
# CPU-only version
./busywork [OPTIONS]

# CUDA version (if available)
./busywork_cuda [OPTIONS]
```

### Options

- `--cpugoal N.NN` - CPU utilization goal 0.0-1.0 (default: (cpucount-1)/cpucount)
- `--gpugoal N.NN` - GPU utilization goal 0.0-1.0 (default: 0.99)
- `--gputhreads N` - Number of GPU worker threads (default: 1)
- `-h, --help` - Show help message

### Examples

```bash
# Run with default settings
./busywork

# Target 80% CPU utilization
./busywork --cpugoal 0.8

# Target 50% CPU and 75% GPU
./busywork_cuda --cpugoal 0.5 --gpugoal 0.75

# Use multiple GPU worker threads
./busywork_cuda --gputhreads 2 --gpugoal 0.9
```

## How It Works

### CPU Heat Generation
- Creates worker threads (default: CPU count - 1)
- Each thread performs intensive floating-point math (sin, cos, sqrt)
- Monitor thread checks CPU load every 5 seconds via `/proc/stat`
- Dynamically activates/deactivates threads to match target utilization

### GPU Heat Generation (CUDA version only)
- Launches CUDA kernels with configurable grid/block sizes
- Performs intensive GPU computations (trig, FMA operations)
- Uses NVML to monitor GPU utilization
- Adjusts duty cycle (work/sleep ratio) to match target load

Press Enter to stop the program gracefully.

## Requirements

### CPU-only version
- g++ with C++17 support
- Linux (uses `/proc/stat` for monitoring)
- pthread

### CUDA version
- All CPU version requirements
- CUDA Toolkit (tested with CUDA 11.0+)
- NVIDIA GPU
- NVML library (nvidia-ml)

## Project Structure

```
busywork/
├── src/
│   ├── main.cpp          # Main program and CPU workers
│   ├── gpu_worker.h      # GPU interface header
│   └── gpu_kernel.cu     # CUDA kernel implementation
├── build/                # Compiled objects
├── backup/               # Source backups
├── logs/                 # Build and run logs
├── Makefile              # Build system
├── README.md             # This file
└── ARCHITECTURE.md       # Technical design document
```
