CXX = g++
NVCC = nvcc
CXXFLAGS = -std=c++17 -pthread -O3 -Wall -Wextra
NVCCFLAGS = -O3 -std=c++14 -ccbin /usr/bin/gcc-12 -Xcudafe --diag_suppress=20011 -Xcudafe --diag_suppress=20014
TARGET = busywork
SRC_DIR = src
BUILD_DIR = build
BACKUP_DIR = backup
LOGS_DIR = logs

# Detect CUDA installation
CUDA_PATH ?= /usr/local/cuda
CUDA_AVAILABLE := $(shell [ -d $(CUDA_PATH) ] && echo yes)

ifeq ($(CUDA_AVAILABLE),yes)
    CXXFLAGS += -DENABLE_CUDA -I$(CUDA_PATH)/include
    LDFLAGS = -L$(CUDA_PATH)/lib64 -lcudart -lnvidia-ml -lpthread -lm
    GPU_OBJECTS = $(BUILD_DIR)/gpu_kernel.o
    BUILD_MSG = "Built with CUDA support"
else
    LDFLAGS = -lpthread -lm
    GPU_OBJECTS =
    BUILD_MSG = "Built CPU-only (CUDA not found)"
endif

MAIN_OBJECT = $(BUILD_DIR)/main.o

all: $(BUILD_DIR) $(BACKUP_DIR) $(LOGS_DIR) $(TARGET)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BACKUP_DIR):
	mkdir -p $(BACKUP_DIR)/src

$(LOGS_DIR):
	mkdir -p $(LOGS_DIR)

$(TARGET): $(MAIN_OBJECT) $(GPU_OBJECTS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)
	@echo $(BUILD_MSG)

$(BUILD_DIR)/main.o: $(SRC_DIR)/main.cpp $(SRC_DIR)/gpu_worker.h
	$(CXX) $(CXXFLAGS) -c -o $@ $<

ifeq ($(CUDA_AVAILABLE),yes)
$(BUILD_DIR)/gpu_kernel.o: $(SRC_DIR)/gpu_kernel.cu $(SRC_DIR)/gpu_worker.h
	$(NVCC) $(NVCCFLAGS) -c -o $@ $<
endif

clean:
	rm -rf $(BUILD_DIR) $(TARGET)

.PHONY: all clean
