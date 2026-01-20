#ifndef BUSY_H
#define BUSY_H

#include <atomic>
#include <memory>
#include <vector>

extern std::atomic<bool> RUNNING;
extern std::atomic<double> gpu_duty_cycle;

void gpu_worker_entry(int id);

#endif
