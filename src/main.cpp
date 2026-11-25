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
#include <iomanip>
#include <algorithm>
#include <utility>
#include <string>

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

inline double cToF( double c ) {
    return c * 9.0 / 5.0 + 32.0;
}

inline std::string formatTemp( double c ) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision( 1 ) << c << "°C (" << cToF( c ) << "°F)";
    return oss.str();
}

struct EnergyMetrics {
    double cpuPower = 0.0;
    bool cpuPowerEstimated = false;
    double cpuTemp = 0.0;
    
    double gpuPower = 0.0;
    double gpuTemp = 0.0;
    
    std::vector<double> amdGpuPowers;
    std::vector<double> amdGpuTemps;
    
    double totalPowerWatts = 0.0;
    double totalHeatBtuPerHour = 0.0;
    
    double cpuHeatPerWatt = 0.0;
    double cpuEfficiencyScore = 0.0;
    
    double gpuHeatPerWatt = 0.0;
    double gpuEfficiencyScore = 0.0;
    
    double amdHeatPerWatt = 0.0;
    double amdEfficiencyScore = 0.0;
    
    double heatPerWattThermal = 1.0; // 1.0 W_heat / W_elec
    double thermodynamicHeatPerWattBtu = 3.412142; // BTU/Wh
    double overallEfficiencyScore = 0.0;
    
    double totalJoules = 0.0;
    double totalBtu = 0.0;
    double totalKwh = 0.0;
};

class EnergyMonitor {
private:
    std::string cpuTempHwmonPath;
    std::string cpuRaplPath;
    std::vector<std::string> amdHwmonPaths;
    
    long long prevRaplEnergyUj = -1;
    std::chrono::steady_clock::time_point prevRaplTime;
    
    double accumulatedJoules = 0.0;
    std::chrono::steady_clock::time_point prevSampleTime;
    bool hasPrevSampleTime = false;
    
    std::string readFileFirstLine( const std::string& path ) {
        std::ifstream file( path );
        if ( !file.is_open() ) return "";
        std::string line;
        std::getline( file, line );
        return line;
    }

    void discoverHwmonSensors() {
        for ( int i = 0; i < 32; ++i ) {
            std::string basePath = "/sys/class/hwmon/hwmon" + std::to_string( i );
            std::string name = readFileFirstLine( basePath + "/name" );
            if ( name.empty() ) continue;
            
            if ( cpuTempHwmonPath.empty() && 
                 ( name == "k10temp" || name == "coretemp" || name == "zenpower" || name == "cpu_thermal" ) ) {
                if ( !readFileFirstLine( basePath + "/temp1_input" ).empty() ) {
                    cpuTempHwmonPath = basePath + "/temp1_input";
                }
            }
            
            if ( name == "amdgpu" ) {
                amdHwmonPaths.push_back( basePath );
            }
        }
        
        std::string raplPath = "/sys/class/powercap/intel-rapl:0/energy_uj";
        std::ifstream testRapl( raplPath );
        if ( testRapl.is_open() ) {
            cpuRaplPath = raplPath;
        }
    }

public:
    EnergyMonitor() {
        discoverHwmonSensors();
    }
    
    double sampleCpuTemperature() {
        if ( cpuTempHwmonPath.empty() ) return 0.0;
        std::string val = readFileFirstLine( cpuTempHwmonPath );
        if ( val.empty() ) return 0.0;
        try {
            return std::stod( val ) / 1000.0;
        } catch ( ... ) {
            return 0.0;
        }
    }
    
    double sampleCpuPower( double currentCpuLoad, int cpuCount, bool& isEstimated ) {
        if ( !cpuRaplPath.empty() ) {
            std::string val = readFileFirstLine( cpuRaplPath );
            if ( !val.empty() ) {
                try {
                    long long energyUj = std::stoll( val );
                    auto now = std::chrono::steady_clock::now();
                    if ( prevRaplEnergyUj >= 0 ) {
                        double deltaSec = std::chrono::duration<double>( now - prevRaplTime ).count();
                        if ( deltaSec > 0.1 ) {
                            long long diffUj = energyUj - prevRaplEnergyUj;
                            if ( diffUj >= 0 ) {
                                double watts = ( diffUj / 1000000.0 ) / deltaSec;
                                prevRaplEnergyUj = energyUj;
                                prevRaplTime = now;
                                isEstimated = false;
                                return watts;
                            }
                        }
                    } else {
                        prevRaplEnergyUj = energyUj;
                        prevRaplTime = now;
                    }
                } catch ( ... ) {}
            }
        }
        
        isEstimated = true;
        double baseIdleWatts = 25.0;
        double maxWatts = std::max( 65.0, cpuCount * 4.5 );
        double loadRatio = ( cpuCount > 0 ) ? ( currentCpuLoad / cpuCount ) : 0.0;
        if ( loadRatio < 0.0 ) loadRatio = 0.0;
        if ( loadRatio > 1.0 ) loadRatio = 1.0;
        return baseIdleWatts + ( maxWatts - baseIdleWatts ) * loadRatio;
    }
    
    void sampleAmdGpu( size_t index, double& powerWatts, double& tempC ) {
        powerWatts = 0.0;
        tempC = 0.0;
        if ( index >= amdHwmonPaths.size() ) return;
        const std::string& basePath = amdHwmonPaths[index];
        
        std::string pStr = readFileFirstLine( basePath + "/power1_input" );
        if ( pStr.empty() ) {
            pStr = readFileFirstLine( basePath + "/power1_average" );
        }
        if ( !pStr.empty() ) {
            try {
                powerWatts = std::stod( pStr ) / 1000000.0;
            } catch ( ... ) {}
        }
        
        std::string tStr = readFileFirstLine( basePath + "/temp1_input" );
        if ( !tStr.empty() ) {
            try {
                tempC = std::stod( tStr ) / 1000.0;
            } catch ( ... ) {}
        }
    }
    
    EnergyMetrics calculateMetrics( double currentCpu, double targetCpu, int cpuCount,
                                   bool gpuAvailable, double gpuLoad, double targetGpu,
                                   const std::vector<std::pair<std::string, double>>& amdWorkersInfo ) {
        EnergyMetrics m;
        auto now = std::chrono::steady_clock::now();
        double deltaSec = 5.0;
        if ( hasPrevSampleTime ) {
            deltaSec = std::chrono::duration<double>( now - prevSampleTime ).count();
            if ( deltaSec <= 0.0 ) deltaSec = 5.0;
        } else {
            hasPrevSampleTime = true;
        }
        prevSampleTime = now;
        
        // CPU
        m.cpuTemp = sampleCpuTemperature();
        m.cpuPower = sampleCpuPower( currentCpu, cpuCount, m.cpuPowerEstimated );
        m.totalPowerWatts += m.cpuPower;
        
        if ( m.cpuPower > 0.0 ) {
            m.cpuHeatPerWatt = currentCpu / m.cpuPower;
        }
        if ( targetCpu > 0.0 ) {
            m.cpuEfficiencyScore = std::min( 100.0, std::max( 0.0, ( currentCpu / targetCpu ) * 100.0 ) );
        } else {
            m.cpuEfficiencyScore = 100.0;
        }
        
        // NVIDIA GPU
        if ( gpuAvailable ) {
#ifdef ENABLE_CUDA
            m.gpuPower = queryGpuPowerWatts();
            m.gpuTemp = queryGpuTemperature();
            m.totalPowerWatts += m.gpuPower;
            
            if ( m.gpuPower > 0.0 ) {
                m.gpuHeatPerWatt = gpuLoad / m.gpuPower;
            }
            double refTarget = ( targetGpu > 0.0 ) ? targetGpu : 1.0;
            m.gpuEfficiencyScore = std::min( 100.0, std::max( 0.0, ( gpuLoad / refTarget ) * 100.0 ) );
#endif
        }
        
        // AMD Vulkan GPUs
        double totalAmdDuty = 0.0;
        double totalAmdPower = 0.0;
        for ( size_t i = 0; i < amdWorkersInfo.size(); ++i ) {
            double pWatts = 0.0;
            double tDeg = 0.0;
            sampleAmdGpu( i, pWatts, tDeg );
            
            if ( pWatts <= 0.001 ) {
                double duty = amdWorkersInfo[i].second;
                pWatts = 10.0 + duty * 25.0;
            }
            
            m.amdGpuPowers.push_back( pWatts );
            m.amdGpuTemps.push_back( tDeg );
            m.totalPowerWatts += pWatts;
            
            totalAmdPower += pWatts;
            totalAmdDuty += amdWorkersInfo[i].second;
        }
        
        if ( !amdWorkersInfo.empty() ) {
            if ( totalAmdPower > 0.0 ) {
                m.amdHeatPerWatt = totalAmdDuty / totalAmdPower;
            }
            double avgDuty = totalAmdDuty / amdWorkersInfo.size();
            m.amdEfficiencyScore = std::min( 100.0, std::max( 0.0, avgDuty * 100.0 ) );
        }
        
        m.totalHeatBtuPerHour = m.totalPowerWatts * 3.412142;
        
        double weightedScoreSum = 0.0;
        double totalWeights = 0.0;
        
        if ( m.cpuPower > 0.0 ) {
            weightedScoreSum += m.cpuEfficiencyScore * m.cpuPower;
            totalWeights += m.cpuPower;
        }
        if ( gpuAvailable && m.gpuPower > 0.0 ) {
            weightedScoreSum += m.gpuEfficiencyScore * m.gpuPower;
            totalWeights += m.gpuPower;
        }
        if ( !amdWorkersInfo.empty() && totalAmdPower > 0.0 ) {
            weightedScoreSum += m.amdEfficiencyScore * totalAmdPower;
            totalWeights += totalAmdPower;
        }
        
        if ( totalWeights > 0.0 ) {
            m.overallEfficiencyScore = weightedScoreSum / totalWeights;
        } else {
            m.overallEfficiencyScore = 100.0;
        }
        
        accumulatedJoules += m.totalPowerWatts * deltaSec;
        m.totalJoules = accumulatedJoules;
        m.totalBtu = accumulatedJoules * 0.000947817;
        m.totalKwh = accumulatedJoules / 3600000.0;
        
        return m;
    }
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
    EnergyMonitor energyMonitor;
    double maxTempCap = 85.0;
    
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
            
            std::vector<std::pair<std::string, double>> amdInfo;
#ifdef ENABLE_VULKAN
            for ( const auto& worker : amdGpuWorkers ) {
                amdInfo.emplace_back( worker->device.name, worker->active ? worker->dutyCycle.load() : 0.0 );
            };
#endif

            double targetGpu = gpuAvailable ? gpuGoal.load() : 0.0;
            EnergyMetrics metrics = energyMonitor.calculateMetrics(
                currentCpuLoad, targetCpuLoad, cpuCount,
                gpuAvailable, gpuLoad, targetGpu,
                amdInfo
            );
            
            printStatus( currentCpuLoad, targetCpuLoad, activeCount, gpuLoad, metrics );
        };
    };
    
    void printStatus( double currentCpu, double targetCpu, int activeCpuThreads, double gpuLoad, const EnergyMetrics& metrics ) {
        std::cout << "[Status]" << std::endl;
        std::cout << "  CPU Load: " << std::fixed << std::setprecision(1) << currentCpu
                  << " / Goal: " << targetCpu << " threads"
                  << " | Active Workers: " << activeCpuThreads << "/" << cpuThreadStates.size();
        if ( metrics.cpuTemp > 0.0 ) {
            std::cout << " | Temp: " << formatTemp( metrics.cpuTemp );
        };
        std::cout << " | Power: " << metrics.cpuPower << " W" 
                  << ( metrics.cpuPowerEstimated ? " (est.)" : "" ) << std::endl;
        
        if ( gpuAvailable ) {
            std::cout << "  GPU Load: " << gpuLoad << " / Goal: " << gpuGoal.load() 
                      << " | Duty Cycle: " << std::setprecision(2) << gpuDutyCycle.load();
            if ( metrics.gpuTemp > 0.0 ) {
                std::cout << " | Temp: " << formatTemp( metrics.gpuTemp );
            };
            std::cout << " | Power: " << std::setprecision(1) << metrics.gpuPower << " W" << std::endl;
        };

#ifdef ENABLE_VULKAN
        for ( size_t i = 0; i < amdGpuWorkers.size(); ++i ) {
            const auto& worker = amdGpuWorkers[i];
            std::cout << "  AMD Vulkan GPU: " << worker->device.name
                      << " | Active: " << ( worker->active ? "Yes" : "No" )
                      << " | Duty Cycle: " << std::setprecision(2) << worker->dutyCycle.load();
            if ( i < metrics.amdGpuTemps.size() && metrics.amdGpuTemps[i] > 0.0 ) {
                std::cout << " | Temp: " << formatTemp( metrics.amdGpuTemps[i] );
            };
            if ( i < metrics.amdGpuPowers.size() && metrics.amdGpuPowers[i] > 0.0 ) {
                std::cout << " | Power: " << std::setprecision(1) << metrics.amdGpuPowers[i] << " W";
            };
            std::cout << std::endl;
        };
#endif
        
        // Check for thermal cap warnings
        std::vector<std::string> thermalWarnings;
        if ( metrics.cpuTemp >= maxTempCap ) {
            thermalWarnings.push_back( "CPU temperature " + formatTemp( metrics.cpuTemp ) + " exceeded cap (" + formatTemp( maxTempCap ) + ")" );
        };
        if ( gpuAvailable && metrics.gpuTemp >= maxTempCap ) {
            thermalWarnings.push_back( "NVIDIA GPU temperature " + formatTemp( metrics.gpuTemp ) + " exceeded cap (" + formatTemp( maxTempCap ) + ")" );
        };
#ifdef ENABLE_VULKAN
        for ( size_t i = 0; i < amdGpuWorkers.size(); ++i ) {
            if ( i < metrics.amdGpuTemps.size() && metrics.amdGpuTemps[i] >= maxTempCap ) {
                thermalWarnings.push_back( "AMD GPU (" + amdGpuWorkers[i]->device.name + ") temperature " + formatTemp( metrics.amdGpuTemps[i] ) + " exceeded cap (" + formatTemp( maxTempCap ) + ")" );
            };
        };
#endif

        if ( !thermalWarnings.empty() ) {
            std::cout << "  >>> [THERMAL WARNING] Hardware temperature cap exceeded! <<<" << std::endl;
            for ( const auto& warn : thermalWarnings ) {
                std::cout << "  >>> [WARNING] " << warn << "!" << std::endl;
            };
        };

        std::cout << "  [Energy Efficiency & Heat Metrics]" << std::endl;
        std::cout << "    Total Thermal Power: " << std::setprecision(1) << metrics.totalPowerWatts << " W"
                  << " (" << metrics.totalHeatBtuPerHour << " BTU/h heat output)" << std::endl;
        
        std::cout << "    CPU Heat/Watt: " << std::setprecision(3) << metrics.cpuHeatPerWatt << " threads/W"
                  << " | Efficiency Score: " << std::setprecision(1) << metrics.cpuEfficiencyScore << "/100" << std::endl;
        
        if ( gpuAvailable ) {
            std::cout << "    GPU Heat/Watt: " << std::setprecision(2) << metrics.gpuHeatPerWatt << " SMs/W"
                      << " | Efficiency Score: " << std::setprecision(1) << metrics.gpuEfficiencyScore << "/100" << std::endl;
        };

#ifdef ENABLE_VULKAN
        if ( !amdGpuWorkers.empty() && metrics.amdHeatPerWatt > 0.0 ) {
            std::cout << "    AMD GPU Heat/Watt: " << std::setprecision(3) << metrics.amdHeatPerWatt << " duty/W"
                      << " | Efficiency Score: " << std::setprecision(1) << metrics.amdEfficiencyScore << "/100" << std::endl;
        };
#endif

        std::cout << "    Thermodynamic Heat-per-Watt: 3.412 BTU/Wh (1.000 W_heat/W_elec)"
                  << " | Overall Heating Score: " << std::setprecision(1) << metrics.overallEfficiencyScore << "/100" << std::endl;
        std::cout << "    Heater Benchmark Comparisons (3.412 BTU/Wh thermal conversion):" << std::endl;
        std::cout << "      - vs. Resistive Electric Space Heater: 100.0% parity (3.412 BTU/Wh)" << std::endl;
        std::cout << "      - vs. Standard Gas Furnace (80% AFUE):  125.0% on-site eff. (vs. 2.730 BTU/Wh-eq)" << std::endl;
        std::cout << "      - vs. High-Eff Condensing Gas (96%):    104.2% on-site eff. (vs. 3.276 BTU/Wh-eq)" << std::endl;
        std::cout << "      - vs. Air-Source Heat Pump (COP 3.0):    33.3% heat moved (vs. 10.236 BTU/Wh)" << std::endl;
        std::cout << "    Cumulative Heat Delivered: " << std::setprecision(1) << metrics.totalJoules / 1000.0 << " kJ"
                  << " (" << metrics.totalBtu << " BTU / " << std::setprecision(4) << metrics.totalKwh << " kWh)" << std::endl;
        
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
    BusyWork( double cpuGoalThreads, double gpuGoalThreads, int gpuThreads = 1, bool enableAmdGpu = true, double maxTemp = 85.0 )
        : cpuMonitor( std::thread::hardware_concurrency() ), maxTempCap( maxTemp > 0.0 ? maxTemp : 85.0 ) {
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
        std::cout << "Energy & Thermal Monitoring: Active (real-time power, temperature, and heat-per-watt efficiency tracking)" << std::endl;
        std::cout << "Max Temperature Cap: " << formatTemp( maxTempCap ) << " (warning threshold)" << std::endl;
        
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
    std::cout << "  -m, --max-temp N     Max temperature cap in °C before warnings (default: 85.0)" << std::endl;
    std::cout << "  -h, --help           Show this help message" << std::endl;
    std::cout << "\nRun without arguments to see system capabilities." << std::endl;
};

int main( int argc, char* argv[] ) {
    double cpuGoal = -1.0;  // -1 means use default
    double gpuGoal = -1.0;  // -1 means use default
    int gpuThreads = 1;
    bool enableAmdGpu = true;
    double maxTemp = 85.0;
    
    static struct option longOptions[] = {
        { "cpugoal", required_argument, 0, 'c' },
        { "gpugoal", required_argument, 0, 'g' },
        { "gputhreads", required_argument, 0, 't' },
        { "no-amd-gpu", no_argument, 0, 'a' },
        { "max-temp", required_argument, 0, 'm' },
        { "tempcap", required_argument, 0, 'm' },
        { "help", no_argument, 0, 'h' },
        { 0, 0, 0, 0 }
    };
    
    int opt;
    int optionIndex = 0;
    
    while ( ( opt = getopt_long( argc, argv, "hc:g:t:m:a", longOptions, &optionIndex ) ) != -1 ) {
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
            case 'm':
                maxTemp = std::stod( optarg );
                break;
            case 'h':
                printUsage( argv[0] );
                return 0;
            default:
                printUsage( argv[0] );
                return 1;
        };
    };
    
    BusyWork busy( cpuGoal, gpuGoal, gpuThreads, enableAmdGpu, maxTemp );
    busy.start();
    busy.waitForSignal();
    
    return 0;
};
