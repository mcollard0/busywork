#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

#include <nvml.h>

#include "busy.h"
#include "busy_cuda.h"

using namespace std;

// --- Globals ---
atomic<bool> RUNNING(true);
atomic<double> gpu_duty_cycle(1.0);

int CPU_THREADS = 0;
int GPU_THREADS = 1;
double CPU_GOAL = -1.0; // interpreted as "number of cores"
double GPU_GOAL = -1.0; // interpreted as "number of GPUs" (0.0 - N.0)

vector<unique_ptr<atomic<bool>>> cpu_thread_states;

// --- CPU Worker ---
void cpu_worker(int id, atomic<bool> *active) {
  while (RUNNING) {
    if (*active) {
      double val = 1.0;
      // Heavy floating point work
      for (int i = 0; i < 100000; ++i) {
        val = sin(val) * tan(val);
        val = sqrt(std::abs(val) + 1.0);
      }
      volatile double sink = val;
      (void)sink;
    } else {
      this_thread::sleep_for(chrono::milliseconds(10));
    }
  }
}

// --- GPU Worker (Host Side) ---
void gpu_worker_entry(int id) {
  // Round robin device assignment
  int device_count = 0;
  get_device_count(&device_count);
  if (device_count > 0) {
    set_device(id % device_count);
  }

  float *d_dummy = gpu_alloc(sizeof(float));

  int block_size = 256;
  int grid_size = 1024;
  int iterations = 100000;

  while (RUNNING) {
    double duty = gpu_duty_cycle.load();

    if (duty > 0.01) {
      launch_gpu_kernel_async(grid_size, block_size, iterations, d_dummy);
      gpu_sync();

      if (duty < 0.99) {
        // Heuristic sleep
        int sleep_us = (int)(1000.0 * (1.0 - duty) * 50.0);
        if (sleep_us > 0)
          this_thread::sleep_for(chrono::microseconds(sleep_us));
      }
    } else {
      this_thread::sleep_for(chrono::milliseconds(50));
    }
  }
  gpu_free(d_dummy);
}

// --- Monitors ---
class CpuMonitor {
  long prev_idle = 0;
  long prev_total = 0;

public:
  double get_cpu_load() {
    ifstream f("/proc/stat");
    string line;
    if (!getline(f, line))
      return 0.0;
    istringstream iss(line);
    string cpu;
    long user, nice, system, idle, iowait, irq, softirq, steal;
    iss >> cpu >> user >> nice >> system >> idle >> iowait >> irq >> softirq >>
        steal;

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
  bool inited = false;
  int device_count = 0;
  vector<nvmlDevice_t> devices;

public:
  GpuMonitor() {
    if (nvmlInit() == NVML_SUCCESS) {
      inited = true;
      unsigned int count = 0;
      if (nvmlDeviceGetCount(&count) == NVML_SUCCESS) {
        device_count = count;
        for (unsigned int i = 0; i < count; ++i) {
          nvmlDevice_t dev;
          if (nvmlDeviceGetHandleByIndex(i, &dev) == NVML_SUCCESS) {
            devices.push_back(dev);
          }
        }
      }
    } else {
      cerr << "Failed to init NVML (Monitor)" << endl;
    }
  }

  ~GpuMonitor() {
    if (inited)
      nvmlShutdown();
  }

  double get_total_gpu_load() {
    if (!inited)
      return 0.0;
    double total = 0.0;
    for (auto &dev : devices) {
      nvmlUtilization_t util;
      if (nvmlDeviceGetUtilizationRates(dev, &util) == NVML_SUCCESS) {
        total += (util.gpu / 100.0);
      }
    }
    return total;
  }

  int get_device_count() { return device_count; }

  string get_device_name(int idx) {
    if (idx < 0 || idx >= devices.size())
      return "Unknown";
    char name[64];
    if (nvmlDeviceGetName(devices[idx], name, sizeof(name)) == NVML_SUCCESS) {
      return string(name);
    }
    return "Unknown";
  }
};

// --- Helpers ---
void print_status(double current_cpu_raw, double target_cpu, double current_gpu,
                  double target_gpu, int active_cpu_threads,
                  int num_cpu_cores) {
  double scaled_cpu = current_cpu_raw * num_cpu_cores;

  cout << "Status Update:" << endl;
  cout << "  CPU: Usage=" << scaled_cpu
       << " cores (Load: " << (current_cpu_raw * 100.0) << "%)"
       << " / Goal=" << target_cpu << " cores"
       << " | Active Threads: " << active_cpu_threads << "/" << CPU_THREADS
       << endl;

  cout << "  GPU: Usage=" << current_gpu << " (sum) / Goal=" << target_gpu
       << " | Duty Cycle: " << gpu_duty_cycle.load() << endl;
  cout << "-------------------------------------------" << endl;
}

int main(int argc, char **argv) {
  unsigned int nproc = std::thread::hardware_concurrency();

  GpuMonitor gpu_mon;
  int gpu_count = gpu_mon.get_device_count();

  cout << "Detected Hardware:" << endl;
  cout << "  CPU Cores (Threads): " << nproc << endl;
  cout << "  GPUs Detected: " << gpu_count << endl;
  for (int i = 0; i < gpu_count; i++) {
    cout << "    GPU " << i << ": " << gpu_mon.get_device_name(i) << endl;
  }
  cout << endl;

  if (argc == 1) {
    cout << "Usage: ./busy --cputhreads <N> --cpugoal <N.N> --gputhread <N> "
            "--gpugoal <N.N>"
         << endl;
    cout << "Running defaults..." << endl << endl;
  }

  CPU_THREADS = (nproc > 1) ? nproc - 1 : 1;
  CPU_GOAL = (double)CPU_THREADS;
  GPU_THREADS = 1;
  GPU_GOAL = 0.99;

  for (int i = 1; i < argc; ++i) {
    string arg = argv[i];
    if (arg == "--cputhreads" && i + 1 < argc) {
      CPU_THREADS = stoi(argv[++i]);
    } else if (arg == "--gputhread") {
      if (i + 1 < argc && isdigit(argv[i + 1][0])) {
        GPU_THREADS = stoi(argv[++i]);
      } else {
        GPU_THREADS = 1;
      }
    } else if (arg == "--cpugoal" && i + 1 < argc) {
      CPU_GOAL = stod(argv[++i]);
    } else if (arg == "--gpugoal" && i + 1 < argc) {
      GPU_GOAL = stod(argv[++i]);
    }
  }

  cout << "Starting Busy:" << endl;
  cout << "  CPU Threads: " << CPU_THREADS << " | Goal Load: " << CPU_GOAL
       << " cores" << endl;
  cout << "  GPU Threads: " << GPU_THREADS << " | Goal Load: " << GPU_GOAL
       << " units" << endl;

  vector<thread> pool;
  for (int i = 0; i < CPU_THREADS; ++i) {
    cpu_thread_states.push_back(make_unique<atomic<bool>>(true));
    pool.emplace_back(cpu_worker, i, cpu_thread_states.back().get());
  }

  vector<thread> gpu_pool;
  for (int i = 0; i < GPU_THREADS; ++i) {
    gpu_pool.emplace_back(gpu_worker_entry, i);
  }

  CpuMonitor cpu_mon;

  this_thread::sleep_for(chrono::seconds(1));
  cpu_mon.get_cpu_load();

  while (RUNNING) {
    this_thread::sleep_for(chrono::seconds(5));

    double current_cpu_raw = cpu_mon.get_cpu_load();
    double current_gpu_sum = gpu_mon.get_total_gpu_load();

    double current_cpu_scaled = current_cpu_raw * nproc;

    // --- CPU Regulation ---
    int active_count = 0;
    for (auto &s : cpu_thread_states)
      if (*s)
        active_count++;

    double cpu_error = current_cpu_scaled - CPU_GOAL;

    if (cpu_error < -0.2) {
      int threads_to_add = (int)(std::abs(cpu_error) * 0.8);
      if (threads_to_add < 1)
        threads_to_add = 1;

      if (active_count < CPU_THREADS) {
        int added = 0;
        for (auto &s : cpu_thread_states) {
          if (!(*s)) {
            *s = true;
            added++;
            if (added >= threads_to_add)
              break;
          }
        }
      }
    } else if (cpu_error > 0.2) {
      int threads_to_remove = (int)(cpu_error * 0.8);
      if (threads_to_remove < 1)
        threads_to_remove = 1;

      if (active_count > 0) {
        int removed = 0;
        for (auto &s : cpu_thread_states) {
          if (*s) {
            *s = false;
            removed++;
            if (removed >= threads_to_remove)
              break;
          }
        }
      }
    }

    // --- GPU Regulation ---
    double duty = gpu_duty_cycle.load();
    if (current_gpu_sum < (GPU_GOAL - 0.1)) {
      duty += 0.05;
    } else if (current_gpu_sum > (GPU_GOAL + 0.1)) {
      duty -= 0.05;
    }
    if (duty > 1.0)
      duty = 1.0;
    if (duty < 0.0)
      duty = 0.0;
    gpu_duty_cycle.store(duty);

    print_status(current_cpu_raw, CPU_GOAL, current_gpu_sum, GPU_GOAL,
                 active_count, nproc);
  }

  for (auto &t : pool)
    t.join();
  for (auto &t : gpu_pool)
    t.join();

  return 0;
}
