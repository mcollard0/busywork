#include <iostream>
#include <iomanip>
#include <thread>
#include <atomic>
#include <vector>
#include <chrono>
#include <cmath>
#include <cstring>
#include <csignal>
#include <fstream>
#include <sstream>
#include <algorithm>

#ifdef USE_CUDA
#include "gpu_worker.h"
#endif

struct Config {
    int cpuThreads = -1;      // -1 means auto (cpucount - 1)
    bool enableGpu = false;
    int gpuIntensity = 10;    // GPU work intensity (1-100)
    double cpuGoal = -1.0;    // -1 means auto
};

static std::atomic<bool> running{true};
static std::atomic<int> activeWorkers{0};
static std::atomic<int> targetWorkers{0};

double getCpuUsage() {
    static long long prevIdle = 0, prevTotal = 0;

    std::ifstream stat("/proc/stat");
    std::string line;
    std::getline(stat, line);

    std::istringstream iss(line);
    std::string cpu;
    long long user, nice, system, idle, iowait, irq, softirq, steal;
    iss >> cpu >> user >> nice >> system >> idle >> iowait >> irq >> softirq >> steal;

    long long idleTime = idle + iowait;
    long long totalTime = user + nice + system + idle + iowait + irq + softirq + steal;

    long long idleDelta = idleTime - prevIdle;
    long long totalDelta = totalTime - prevTotal;

    prevIdle = idleTime;
    prevTotal = totalTime;

    if (totalDelta == 0) return 0.0;
    return 100.0 * (1.0 - (double)idleDelta / (double)totalDelta);
}

void cpuWorker(int id) {
    volatile double result = 0.0;
    while (running) {
        if (id < targetWorkers.load()) {
            // Busy work: compute-intensive floating point operations
            for (int i = 0; i < 10000; ++i) {
                result += std::sin(result + i) * std::cos(result - i);
                result = std::fmod(result, 1000000.0);
            }
        } else {
            // This worker should be idle
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }
}

void gpuWorkerThread(int intensity) {
#ifdef USE_CUDA
    while (running) {
        gpuLaunchWork(intensity);
        gpuSync();
    }
#endif
}

void printUsage(const char* prog) {
    std::cout << "Usage: " << prog << " [options]\n"
              << "Options:\n"
              << "  --cputhreads N   Number of CPU worker threads (default: cpucount-1)\n"
              << "  --cpugoal N.NN   Target CPU utilization 0.0-100.0 (default: ~100% of threads)\n"
#ifdef USE_CUDA
              << "  --gpu            Enable GPU busy work\n"
              << "  --gpuintensity N GPU work intensity 1-100 (default: 10)\n"
#endif
              << "  --help           Show this help\n";
}

void signalHandler(int sig) {
    std::cout << "\nShutting down...\n";
    running = false;
}

Config parseArgs(int argc, char* argv[]) {
    Config cfg;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--cputhreads") == 0 && i + 1 < argc) {
            cfg.cpuThreads = std::stoi(argv[++i]);
        } else if (strcmp(argv[i], "--cpugoal") == 0 && i + 1 < argc) {
            cfg.cpuGoal = std::stod(argv[++i]);
        } else if (strcmp(argv[i], "--gpu") == 0) {
            cfg.enableGpu = true;
        } else if (strcmp(argv[i], "--gpuintensity") == 0 && i + 1 < argc) {
            cfg.gpuIntensity = std::stoi(argv[++i]);
            cfg.enableGpu = true;
        } else if (strcmp(argv[i], "--help") == 0) {
            printUsage(argv[0]);
            exit(0);
        }
    }

    return cfg;
}

int main(int argc, char* argv[]) {
    std::signal(SIGINT, signalHandler);
    std::signal(SIGTERM, signalHandler);

    Config cfg = parseArgs(argc, argv);

    int cpuCount = std::thread::hardware_concurrency();
    int maxThreads = (cfg.cpuThreads > 0) ? cfg.cpuThreads : (cpuCount - 1);
    maxThreads = std::max(1, maxThreads);

    double goalUtilization = (cfg.cpuGoal >= 0) ? cfg.cpuGoal : (100.0 * maxThreads / cpuCount);
    goalUtilization = std::clamp(goalUtilization, 0.0, 100.0);

    std::cout << "Busy - CPU/GPU Heat Generator\n"
              << "CPU cores: " << cpuCount << "\n"
              << "Max worker threads: " << maxThreads << "\n"
              << "Target CPU utilization: " << goalUtilization << "%\n";

#ifdef USE_CUDA
    bool gpuActive = false;
    std::thread gpuThread;

    if (cfg.enableGpu) {
        if (gpuAvailable()) {
            if (gpuInit(0)) {
                gpuActive = true;
                std::cout << "GPU intensity: " << cfg.gpuIntensity << "\n";
                gpuThread = std::thread(gpuWorkerThread, cfg.gpuIntensity);
            }
        } else {
            std::cout << "GPU requested but no CUDA devices found\n";
        }
    }
#else
    if (cfg.enableGpu) {
        std::cout << "GPU support not compiled in (rebuild with CUDA)\n";
    }
#endif

    std::cout << "Press Ctrl+C to stop\n\n";

    // Start with all workers
    targetWorkers.store(maxThreads);

    std::vector<std::thread> workers;
    for (int i = 0; i < maxThreads; ++i) {
        workers.emplace_back(cpuWorker, i);
    }

    // Prime the CPU usage reading
    getCpuUsage();
    std::this_thread::sleep_for(std::chrono::milliseconds(500));

    // Control loop - adjust every 5 seconds
    while (running) {
        std::this_thread::sleep_for(std::chrono::seconds(5));

        double currentUsage = getCpuUsage();
        int current = targetWorkers.load();

        std::cout << "CPU: " << std::fixed << std::setprecision(1) << currentUsage
                  << "% (goal: " << goalUtilization << "%, workers: " << current << "/" << maxThreads << ")\n";

        // Adjust workers based on utilization difference
        double diff = goalUtilization - currentUsage;

        if (diff > 5.0 && current < maxThreads) {
            // Need more work
            int add = std::max(1, (int)(diff / 10.0));
            targetWorkers.store(std::min(maxThreads, current + add));
        } else if (diff < -5.0 && current > 1) {
            // Too much work
            int remove = std::max(1, (int)(-diff / 10.0));
            targetWorkers.store(std::max(1, current - remove));
        }
    }

    // Cleanup
    running = false;

#ifdef USE_CUDA
    if (gpuActive) {
        gpuThread.join();
        gpuCleanup();
    }
#endif

    for (auto& t : workers) {
        t.join();
    }

    return 0;
}
