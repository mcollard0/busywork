#include <iostream>
#include <vector>
#include <thread>
#include <atomic>
#include <chrono>
#include <cmath>
#include <string>
#include <cstring>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <mutex>

#include <cuda_runtime.h>
#include <nvml.h>
#include <unistd.h>

using namespace std;

// --- Globals & Config ---
int CPU_THREADS = 0;
int GPU_THREADS = 0; // Host threads launching GPU work
double CPU_GOAL = -1.0; // 0.0 to 1.0 (defaults to max)
double GPU_GOAL = -1.0; // 0.0 to 1.0 (defaults to max)

bool RUNNING = true;

// CPU Control
vector<unique_ptr<atomic<bool>>> cpu_thread_states;

// GPU Control
// For simplicity, we assume 1 GPU or all GPUs. Let's target GPU 0 for now or all visible.
// We will use a duty cycle approach: Work time vs Sleep time.
atomic<double> gpu_duty_cycle(1.0); // 0.0 to 1.0. 1.0 = Max, 0.0 = Sleep.

// --- CPU Worker ---
void cpu_worker(int id, atomic<bool>* active) {
    while (RUNNING) {
        if (*active) {
            // Heavy floating point work
            double val = 1.0;
            for (int i = 0; i < 100000; ++i) {
                val = sin(val) * tan(val);
                val = sqrt(std::abs(val) + 1.0);
            }
            // Use volatile to prevent optimization (though the loop dependency usually suffices)
            volatile double sink = val; 
            (void)sink;
        } else {
            this_thread::sleep_for(chrono::milliseconds(10));
        }
    }
}

// --- GPU Kernel ---
__global__ void busy_kernel(int iterations, float* dummy_out) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float val = 1.0f + idx * 0.0001f;
    
    // Heavy ALU loop
    for (int i = 0; i < iterations; ++i) {
        val = sinf(val) * cosf(val);
        val = sqrtf(fabsf(val) + 1.0f);
        val = fmaf(val, val, 1.0f);
    }
    
    if (idx == 0) *dummy_out = val; // Prevent optimization
}

// --- GPU Worker ---
void gpu_worker(int id) {
    // Allocate device memory once
    float* d_dummy;
    cudaMalloc(&d_dummy, sizeof(float));
    
    int block_size = 256;
    int grid_size = 1024; 
    // Kernel duration tuning:
    // We want the kernel to last maybe 10-50ms so we can control duty cycle.
    int iterations = 100000; 

    while (RUNNING) {
        double duty = gpu_duty_cycle.load();

        if (duty > 0.01) {
            // Launch kernel
            busy_kernel<<<grid_size, block_size>>>(iterations, d_dummy);
            cudaDeviceSynchronize();
            
            // If duty < 1.0, we need to sleep to reduce load
            // Duty = Work / (Work + Sleep)
            // Sleep = Work * (1/Duty - 1)
            // But we don't know exact 'Work' time easily without measuring.
            // Simplified approach: burst wait.
            // Actually, if we just run continuously, we get 100%. 
            // If we want less, we sleep.
            // Let's rely on the controller to set 'duty' which here functionally acts as a probability or pause.
            // Better: Run batch, then sleep proportional to (1-duty).
            
            // To make it smoother:
            // Always run work.
            // Then sleep for K * (1 - duty).
            // If duty is 1.0, sleep is 0.
            // If duty is 0.5, sleep should equal work time.
            // If duty is 0.1, sleep should be 9x work time.
            
            // Let's assume constant work time for now or small sleeps.
            // We blindly sleep somewhat based on the inverse.
            
            if (duty < 0.99) {
                // Heuristic sleep.
                // Determine sleep duration in microseconds based on an arbitrary base "period"
                // Let's say we want a cycle of 50ms.
                // Work takes W. Sleep takes S.
                // We just insert a sleep. 
                int sleep_us = (int)(1000.0 * (1.0 - duty) * 50.0); 
                // Scaling factor 50.0 is arbitrary "aggressiveness" of throttling.
                // If duty is low, we sleep more.
                if (sleep_us > 0)
                     this_thread::sleep_for(chrono::microseconds(sleep_us));
            }
            
        } else {
            // Duty effective 0
            this_thread::sleep_for(chrono::milliseconds(50));
        }
    }
    cudaFree(d_dummy);
}

// --- Monitors ---
class CpuMonitor {
    long prev_idle = 0;
    long prev_total = 0;
public:
    double get_cpu_load() {
        ifstream f("/proc/stat");
        string line;
        getline(f, line); // cpu ...
        istringstream iss(line);
        string cpu;
        long user, nice, system, idle, iowait, irq, softirq, steal;
        iss >> cpu >> user >> nice >> system >> idle >> iowait >> irq >> softirq >> steal;
        
        long total = user + nice + system + idle + iowait + irq + softirq + steal;
        long total_idle = idle + iowait;
        
        long diff_idle = total_idle - prev_idle;
        long diff_total = total - prev_total;
        
        double load = 0.0;
        if (diff_total > 0) {
            load = (double)(diff_total - diff_idle) / diff_total;
        }
        
        prev_idle = total_idle;
        prev_total = total;
        return load;
    }
};

class GpuMonitor {
    nvmlDevice_t device;
    bool inited = false;
public:
    GpuMonitor() {
        if (nvmlInit() == NVML_SUCCESS) {
            if (nvmlDeviceGetHandleByIndex(0, &device) == NVML_SUCCESS) {
                inited = true;
            } else {
                cerr << "Failed to get NVML handle for GPU 0" << endl;
            }
        } else {
            cerr << "Failed to init NVML" << endl;
        }
    }
    
    ~GpuMonitor() {
        if (inited) nvmlShutdown();
    }
    
    double get_gpu_load() {
        if (!inited) return 0.0;
        nvmlUtilization_t util;
        if (nvmlDeviceGetUtilizationRates(device, &util) == NVML_SUCCESS) {
            return util.gpu / 100.0;
        }
        return 0.0;
    }
};

// --- Helpers ---
void print_status(double current_cpu, double target_cpu, double current_gpu, double target_gpu, int active_cpu_threads) {
    cout << "Status Update:" << endl;
    cout << "  CPU: Load=" << (current_cpu * 100.0) << "% / Goal=" << (target_cpu * 100.0) << "%"
         << " | Active Threads: " << active_cpu_threads << "/" << CPU_THREADS << endl;
    
    cout << "  GPU: Load=" << (current_gpu * 100.0) << "% / Goal=" << (target_gpu * 100.0) << "%"
         << " | Duty Cycle: " << gpu_duty_cycle.load() << endl;
    cout << "-------------------------------------------" << endl;
}

int main(int argc, char** argv) {
    // Defaults
    unsigned int nproc = std::thread::hardware_concurrency();
    CPU_THREADS = (nproc > 1) ? nproc - 1 : 1;
    CPU_GOAL = (double)(CPU_THREADS) / nproc; // Implicit default goal: saturate the threads we launch
    // Wait, user said "default: cpucount-1", "Default goal similar".
    // If I launch N-1 threads, max load is (N-1)/N.
    
    GPU_THREADS = 1;
    GPU_GOAL = 0.99;

    // Parse Args
    for (int i = 1; i < argc; ++i) {
        string arg = argv[i];
        if (arg == "--cputhreads" && i+1 < argc) {
            CPU_THREADS = stoi(argv[++i]);
        } else if (arg == "--gputhread") {
            // User: "Busy --gputhread" (maybe flag) or with count?
            // "Busy --gputhread # gpu computation work"
            // Let's assume flag means enable or count if followed by number?
            // User prompt: "Busy --gputhread # gpu computation work"
            // Let's assume it takes an arg if it looks like a number, else default 1.
            if (i+1 < argc && isdigit(argv[i+1][0])) {
                GPU_THREADS = stoi(argv[++i]);
            } else {
                GPU_THREADS = 1;
            }
        } else if (arg == "--cpugoal" && i+1 < argc) {
            CPU_GOAL = stod(argv[++i]);
        } else if (arg == "--gpugoal" && i+1 < argc) {
            GPU_GOAL = stod(argv[++i]);
        }
    }
    
    // Validate Goals
    if (CPU_GOAL > 1.0) CPU_GOAL = 1.0;
    if (GPU_GOAL > 1.0) GPU_GOAL = 1.0;
    if (CPU_GOAL < 0.0) CPU_GOAL = static_cast<double>(CPU_THREADS) / nproc; 
    // Wait, if user didn't specify goal, stick to max potential of launched threads?
    // User said: "--cpugoal #.## (default: cpucount-1)" -> implies goal is thread count? Or utilization?
    // "Default: cpucount-1" implies they want to peg those threads.
    
    cout << "Starting Busy:" << endl;
    cout << "  CPU Threads: " << CPU_THREADS << " | Goal Load: " << CPU_GOAL << endl;
    cout << "  GPU Threads: " << GPU_THREADS << " | Goal Load: " << GPU_GOAL << endl;
    
    // Launch CPU
    vector<thread> pool;
    for (int i = 0; i < CPU_THREADS; ++i) {
        cpu_thread_states.push_back(make_unique<atomic<bool>>(true)); // Start active
        pool.emplace_back(cpu_worker, i, cpu_thread_states.back().get());
    }
    
    // Launch GPU
    vector<thread> gpu_pool;
    for (int i = 0; i < GPU_THREADS; ++i) {
        gpu_pool.emplace_back(gpu_worker, i);
    }
    
    // Monitors
    CpuMonitor cpu_mon;
    GpuMonitor gpu_mon;
    
    // Settle time
    this_thread::sleep_for(chrono::seconds(1));
    cpu_mon.get_cpu_load(); // Prime
    
    // Control Loop
    while (RUNNING) {
        this_thread::sleep_for(chrono::seconds(5));
        
        double current_cpu = cpu_mon.get_cpu_load();
        double current_gpu = gpu_mon.get_gpu_load();
        
        // --- CPU Regulation ---
        // If Load < Goal, activate more threads.
        // If Load > Goal, deactivate threads.
        // We can count current active.
        int active_count = 0;
        for (auto& s : cpu_thread_states) if (*s) active_count++;
        
        if (current_cpu < (CPU_GOAL - 0.02)) {
             // Need more heat
             if (active_count < CPU_THREADS) {
                 // Find an inactive one and wake it
                 for (auto& s : cpu_thread_states) {
                     if (!(*s)) { *s = true; break; } // Wake one by one or more?
                 }
             }
        } else if (current_cpu > (CPU_GOAL + 0.02)) {
            // Too much heat
            if (active_count > 0) {
                // Find an active one and sleep it
                for (auto& s : cpu_thread_states) {
                    if (*s) { *s = false; break; }
                }
            }
        }
        active_count = 0; // recalc for display
        for (auto& s : cpu_thread_states) if (*s) active_count++;
        
        // --- GPU Regulation ---
        // Adjust duty cycle
        double duty = gpu_duty_cycle.load();
        if (current_gpu < (GPU_GOAL - 0.05)) {
            duty += 0.05;
        } else if (current_gpu > (GPU_GOAL + 0.05)) {
            duty -= 0.05;
        }
        if (duty > 1.0) duty = 1.0;
        if (duty < 0.0) duty = 0.0;
        gpu_duty_cycle.store(duty);
        
        print_status(current_cpu, CPU_GOAL, current_gpu, GPU_GOAL, active_count);
    }
    
    // Join (never reached usually)
    for (auto& t : pool) t.join();
    for (auto& t : gpu_pool) t.join();
    
    return 0;
}
