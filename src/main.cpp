#include <iostream>
#include <thread>
#include <vector>
#include <atomic>
#include <chrono>
#include <cmath>
#include <getopt.h>
#include <fstream>
#include <memory>
#include <sstream>

#ifdef ENABLE_CUDA
#include "gpu_worker.h"
#endif

#ifdef ENABLE_VULKAN
#include "amd_vulkan_worker.h"
#endif

class CpuMonitor {
private:
    long prevIdle = 0;
    long prevTotal = 0;
    int cpuCount;
    
public:
    CpuMonitor( int numCpus ) : cpuCount( numCpus ) {};
    
    // Returns CPU load as "number of busy threads" (like load average)
    double getCpuLoad() {
        std::ifstream statFile( "/proc/stat" );
        if ( !statFile.is_open() ) return 0.0;
        
        std::string line;
        std::getline( statFile, line );
        std::istringstream iss( line );
        
        std::string cpu;
        long user, nice, system, idle, iowait, irq, softirq, steal;
        iss >> cpu >> user >> nice >> system >> idle >> iowait >> irq >> softirq >> steal;
        
        long total = user + nice + system + idle + iowait + irq + softirq + steal;
        long totalIdle = idle + iowait;
        
        long diffIdle = totalIdle - prevIdle;
        long diffTotal = total - prevTotal;
        
        double utilization = 0.0;
        if ( diffTotal > 0 ) {
            utilization = static_cast<double>( diffTotal - diffIdle ) / diffTotal;
        };
        
        prevIdle = totalIdle;
        prevTotal = total;
        
        // Convert to "threads worth of load" (0.0 to cpuCount)
        return utilization * cpuCount;
    };
};

class BusyWork {
private:
    std::atomic<bool> running{ true };
    std::atomic<double> cpuGoal;
    std::atomic<double> gpuGoal;
    std::vector<std::thread> cpuThreads;
    std::vector<std::unique_ptr<std::atomic<bool>>> cpuThreadStates;
    std::thread monitorThread;
    std::vector<std::thread> gpuThreads;
    std::atomic<double> gpuDutyCycle{ 1.0 };

#ifdef ENABLE_VULKAN
    struct AmdGpuWorker {
        explicit AmdGpuWorker( AmdVulkanDevice selectedDevice ) : device( std::move( selectedDevice ) ) {};
        AmdVulkanDevice device;
        std::atomic<bool> active{ false };
        std::atomic<double> dutyCycle{ 1.0 };
    };
    std::vector<std::thread> amdGpuThreads;
    std::vector<AmdVulkanDevice> amdGpuDevices;
    std::vector<std::unique_ptr<AmdGpuWorker>> amdGpuWorkers;
    bool amdGpuEnabled = false;
#endif
    
    int cpuCount;
    int gpuThreadCount;
    int gpuCoreCount;
    bool gpuAvailable;
    CpuMonitor cpuMonitor;
    
    // CPU intensive work
    void cpuWork( int threadId, std::atomic<bool>* active ) {
        ( void )threadId;
        double result = 0.0;
        while ( running ) {
            if ( *active ) {
                // Perform mathematical operations to generate heat
                for ( int i = 0; i < 100000; ++i ) {
                    result += std::sin( i ) * std::cos( i );
                    result = std::sqrt( std::abs( result ) );
                };
                volatile double sink = result;
                ( void )sink;
            } else {
                std::this_thread::sleep_for( std::chrono::milliseconds( 10 ) );
            };
        };
    };
    
    // Monitor and adjust workload
    void monitor() {
        std::this_thread::sleep_for( std::chrono::seconds( 1 ) );
        cpuMonitor.getCpuLoad();  // Prime the monitor
        
        while ( running ) {
            std::this_thread::sleep_for( std::chrono::seconds( 5 ) );
            
            double currentCpuLoad = cpuMonitor.getCpuLoad();
            double targetCpuLoad = cpuGoal.load();
            
            // Count active CPU threads
            int activeCount = 0;
            for ( const auto& state : cpuThreadStates ) {
                if ( *state ) activeCount++;
            };
            
            // Adjust CPU thread activity (goal is in "threads worth")
            // Hysteresis of 0.5 threads
            if ( currentCpuLoad < ( targetCpuLoad - 0.5 ) ) {
                // Need more heat - activate a thread
                if ( activeCount < static_cast<int>( cpuThreadStates.size() ) ) {
                    for ( auto& state : cpuThreadStates ) {
                        if ( !( *state ) ) {
                            *state = true;
                            activeCount++;
                            break;
                        };
                    };
                };
            } else if ( currentCpuLoad > ( targetCpuLoad + 0.5 ) ) {
                // Too much heat - deactivate a thread
                if ( activeCount > 0 ) {
                    for ( auto& state : cpuThreadStates ) {
                        if ( *state ) {
                            *state = false;
                            activeCount--;
                            break;
                        };
                    };
                };
            };
            
            // GPU adjustment if available
            double gpuLoad = 0.0;
            if ( gpuAvailable ) {
#ifdef ENABLE_CUDA
                gpuLoad = getGpuLoad();
                double targetGpuLoad = gpuGoal.load();
                double duty = gpuDutyCycle.load();
                
                if ( gpuLoad < ( targetGpuLoad - 0.5 ) ) {
                    duty += 0.05;
                } else if ( gpuLoad > ( targetGpuLoad + 0.5 ) ) {
                    duty -= 0.05;
                };
                
                if ( duty > 1.0 ) duty = 1.0;
                if ( duty < 0.0 ) duty = 0.0;
                gpuDutyCycle.store( duty );
#endif
            };
            
            printStatus( currentCpuLoad, targetCpuLoad, activeCount, gpuLoad );
        };
    };
    
    void printStatus( double currentCpu, double targetCpu, int activeCpuThreads, double gpuLoad ) {
        std::cout << "[Status]" << std::endl;
        std::cout << "  CPU Load: " << currentCpu << " / Goal: " << targetCpu << " threads"
                  << " | Active Workers: " << activeCpuThreads << "/" << cpuThreadStates.size() << std::endl;
        
        if ( gpuAvailable ) {
            std::cout << "  GPU Load: " << gpuLoad << " / Goal: " << gpuGoal.load() 
                      << " | Duty Cycle: " << gpuDutyCycle.load() << std::endl;
        };

#ifdef ENABLE_VULKAN
        for ( const auto& worker : amdGpuWorkers ) {
            std::cout << "  AMD Vulkan GPU: " << worker->device.name
                      << " | Active: " << ( worker->active ? "Yes" : "No" )
                      << " | Duty Cycle: " << worker->dutyCycle.load() << std::endl;
        };
#endif
        
        std::cout << "---" << std::endl;
    };
    
#ifdef ENABLE_CUDA
    void gpuWorker( int threadId ) {
        ( void )threadId;
        GpuContext* ctx = initGpu();
        if ( !ctx ) {
            std::cerr << "Failed to initialize GPU for thread " << threadId << std::endl;
            return;
        };
        
        const int blockSize = 256;
        const int gridSize = 1024;
        const int iterations = 100000;
        
        while ( running ) {
            double duty = gpuDutyCycle.load();
            
            if ( duty > 0.01 ) {
                launchGpuKernel( ctx, gridSize, blockSize, iterations );
                
                if ( duty < 0.99 ) {
                    int sleepMs = static_cast<int>( 50.0 * ( 1.0 - duty ) );
                    if ( sleepMs > 0 ) {
                        std::this_thread::sleep_for( std::chrono::milliseconds( sleepMs ) );
                    };
                };
            } else {
                std::this_thread::sleep_for( std::chrono::milliseconds( 50 ) );
            };
        };

        cleanupGpu( ctx );
    };
    
    double getGpuLoad() {
        return queryGpuLoad();
    };
#endif

#ifdef ENABLE_VULKAN
    void amdGpuWorker( AmdGpuWorker* worker ) {
        AmdVulkanContext* context = initAmdVulkan( worker->device.physicalDeviceIndex );
        if ( !context ) {
            std::cerr << "Failed to initialize AMD Vulkan GPU: " << worker->device.name << std::endl;
            return;
        };

        const int blockSize = 256;
        const int gridSize = 1024;
        const int iterations = 100000;
        worker->active = true;

        while ( running ) {
            double duty = worker->dutyCycle.load();

            if ( duty > 0.01 ) {
                if ( !launchAmdVulkanKernel( context, gridSize, blockSize, iterations ) ) {
                    std::cerr << "AMD Vulkan GPU worker stopped after a failed dispatch: " << worker->device.name << std::endl;
                    break;
                };

                if ( duty < 0.99 ) {
                    int sleepMs = static_cast<int>( 50.0 * ( 1.0 - duty ) );
                    if ( sleepMs > 0 ) {
                        std::this_thread::sleep_for( std::chrono::milliseconds( sleepMs ) );
                    };
                };
            } else {
                std::this_thread::sleep_for( std::chrono::milliseconds( 50 ) );
            };
        };

        worker->active = false;
        cleanupAmdVulkan( context );
    };
#endif

public:
    BusyWork( double cpuGoalThreads, double gpuGoalThreads, int gpuThreads = 1, bool enableAmdGpu = true )
        : cpuMonitor( std::thread::hardware_concurrency() ) {
        cpuCount = std::thread::hardware_concurrency();
        gpuThreadCount = gpuThreads;
        
        // CPU goal is "number of threads worth of load" (like load average)
        if ( cpuGoalThreads > 0 ) {
            cpuGoal = cpuGoalThreads;
        } else {
            // Default: cpucount-1 threads
            cpuGoal = static_cast<double>( cpuCount - 1 );
        };
        
        // Query GPU capabilities
        gpuAvailable = false;
        gpuCoreCount = 0;
        
#ifdef ENABLE_CUDA
        GpuContext* testCtx = initGpu();
        if ( testCtx ) {
            gpuAvailable = true;
            gpuCoreCount = queryGpuCoreCount();
            cleanupGpu( testCtx );
        };
#endif

#ifdef ENABLE_VULKAN
        amdGpuEnabled = enableAmdGpu;
        if ( amdGpuEnabled ) {
            amdGpuDevices = enumerateAmdVulkanDevices();
        };
#else
        ( void )enableAmdGpu;
#endif
        
        // GPU goal is "number of SMs worth of load"
        if ( gpuGoalThreads > 0 ) {
            gpuGoal = gpuGoalThreads;
        } else if ( gpuAvailable && gpuCoreCount > 0 ) {
            gpuGoal = static_cast<double>( gpuCoreCount - 1 );
        } else {
            gpuGoal = 1.0;
        };
    };
    
    void printCapabilities() {
        std::cout << "=== System Capabilities ===" << std::endl;
        std::cout << "CPU Threads: " << cpuCount << std::endl;
        
        if ( gpuAvailable ) {
            std::cout << "GPU Available: Yes" << std::endl;
            std::cout << "GPU Multiprocessors (SMs): " << gpuCoreCount << std::endl;
        } else {
            std::cout << "GPU Available: No" << std::endl;
        };

#ifdef ENABLE_VULKAN
        if ( amdGpuEnabled ) {
            std::cout << "AMD Vulkan GPUs: " << amdGpuDevices.size() << std::endl;
            for ( const auto& device : amdGpuDevices ) {
                std::cout << "  - " << device.name << " (physical device " << device.physicalDeviceIndex << ")" << std::endl;
            };
        } else {
            std::cout << "AMD Vulkan GPUs: Disabled" << std::endl;
        };
#endif
        
        std::cout << "\n=== Current Configuration ===" << std::endl;
        std::cout << "CPU Goal: " << cpuGoal.load() << " threads" << std::endl;
        std::cout << "GPU Goal: " << gpuGoal.load() << " (duty units)" << std::endl;
        std::cout << "GPU Worker Threads: " << gpuThreadCount << std::endl;
        std::cout << "=========================" << std::endl;
    };
    
    ~BusyWork() {
        stop();
    };
    
    void start() {
        printCapabilities();
        std::cout << "\nStarting workload..." << std::endl;
        
        // Start CPU threads - create max threads, control via active flags
        int threadsToStart = ( cpuCount > 1 ) ? ( cpuCount - 1 ) : 1;
        for ( int i = 0; i < threadsToStart; ++i ) {
            cpuThreadStates.push_back( std::make_unique<std::atomic<bool>>( true ) );
            cpuThreads.emplace_back( &BusyWork::cpuWork, this, i, cpuThreadStates.back().get() );
        };
        
        std::cout << "Started " << threadsToStart << " CPU worker threads" << std::endl;
        
        // Start GPU threads if available
        if ( gpuAvailable && gpuThreadCount > 0 ) {
#ifdef ENABLE_CUDA
            for ( int i = 0; i < gpuThreadCount; ++i ) {
                gpuThreads.emplace_back( &BusyWork::gpuWorker, this, i );
            };
            std::cout << "Started " << gpuThreadCount << " GPU worker threads" << std::endl;
#endif
        } else {
            std::cout << "GPU workers: Disabled (no GPU available)" << std::endl;
        };

#ifdef ENABLE_VULKAN
        if ( amdGpuEnabled ) {
            for ( const auto& device : amdGpuDevices ) {
                amdGpuWorkers.push_back( std::make_unique<AmdGpuWorker>( device ) );
                amdGpuThreads.emplace_back( &BusyWork::amdGpuWorker, this, amdGpuWorkers.back().get() );
            };
            if ( !amdGpuThreads.empty() ) {
                std::cout << "Started " << amdGpuThreads.size() << " AMD Vulkan GPU worker threads" << std::endl;
            };
        };
#endif
        
        // Start monitor thread
        monitorThread = std::thread( &BusyWork::monitor, this );
    };
    
    void stop() {
        running = false;
        
        for ( auto& thread : cpuThreads ) {
            if ( thread.joinable() ) {
                thread.join();
            };
        };
        
        for ( auto& thread : gpuThreads ) {
            if ( thread.joinable() ) {
                thread.join();
            };
        };

#ifdef ENABLE_VULKAN
        for ( auto& thread : amdGpuThreads ) {
            if ( thread.joinable() ) {
                thread.join();
            };
        };
#endif
        
        if ( monitorThread.joinable() ) {
            monitorThread.join();
        };
    };
    
    void waitForSignal() {
        std::cout << "Press Enter to stop..." << std::endl;
        std::cin.get();
        stop();
    };
};

void printUsage( const char* programName ) {
    std::cout << "Usage: " << programName << " [OPTIONS]" << std::endl;
    std::cout << "\nGenerate CPU and GPU heat through controlled computational workload." << std::endl;
    std::cout << "\nOptions:" << std::endl;
    std::cout << "  --cpugoal N          CPU load goal in threads (default: cpucount-1)" << std::endl;
    std::cout << "  --gpugoal N          GPU load goal in SMs/cores (default: smcount-1)" << std::endl;
    std::cout << "  --gputhreads N       Number of GPU worker threads (default: 1)" << std::endl;
    std::cout << "  --no-amd-gpu         Disable automatic AMD Vulkan compute workers" << std::endl;
    std::cout << "  -h, --help           Show this help message" << std::endl;
    std::cout << "\nRun without arguments to see system capabilities." << std::endl;
};

int main( int argc, char* argv[] ) {
    double cpuGoal = -1.0;  // -1 means use default
    double gpuGoal = -1.0;  // -1 means use default
    int gpuThreads = 1;
    bool enableAmdGpu = true;
    
    static struct option longOptions[] = {
        { "cpugoal", required_argument, 0, 'c' },
        { "gpugoal", required_argument, 0, 'g' },
        { "gputhreads", required_argument, 0, 't' },
        { "no-amd-gpu", no_argument, 0, 'a' },
        { "help", no_argument, 0, 'h' },
        { 0, 0, 0, 0 }
    };
    
    int opt;
    int optionIndex = 0;
    
    while ( ( opt = getopt_long( argc, argv, "hc:g:t:", longOptions, &optionIndex ) ) != -1 ) {
        switch ( opt ) {
            case 'c':
                cpuGoal = std::stod( optarg );
                break;
            case 'g':
                gpuGoal = std::stod( optarg );
                break;
            case 't':
                gpuThreads = std::stoi( optarg );
                break;
            case 'a':
                enableAmdGpu = false;
                break;
            case 'h':
                printUsage( argv[0] );
                return 0;
            default:
                printUsage( argv[0] );
                return 1;
        };
    };
    
    BusyWork busy( cpuGoal, gpuGoal, gpuThreads, enableAmdGpu );
    busy.start();
    busy.waitForSignal();
    
    return 0;
};
