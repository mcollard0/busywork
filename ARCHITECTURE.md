I wish to introduce heat to my office by way of my cpu and gpu. 

Program: 

Busy --cputhreads # by computational work. Busy --gputhread # gpu computation work. Attempt to busy cpucount-1 threads. 

--cpugoal #.## (default: cpucount-1) --gpugoal #.## (default: similar, processing units - 1)

GPU backends:
- CUDA/NVML targets NVIDIA GPU 0 for existing CUDA workload and utilization control.
- Vulkan automatically enumerates all AMD physical devices with compute queues and starts one worker per detected device, independent of device ordering.
- `--no-amd-gpu` disables all automatic AMD Vulkan workers.
- AMD Vulkan workers use a SPIR-V floating-point kernel and a fixed duty cycle because this program has no cross-vendor AMD utilization telemetry.
- Failure to initialize one AMD Vulkan device does not prevent CPU, CUDA, or other AMD Vulkan workers from running.

Revise execution loop on a regular basis say every 5 seconds, to add or remove work to suit goal utilization.

Language: c++

