NVCC = /usr/local/cuda-13.0/bin/nvcc -ccbin /usr/bin/g++-14 --allow-unsupported-compiler
HOST_COMPILER = /usr/bin/g++-14
CFLAGS = -O3 -std=c++14 -I/usr/local/cuda-13.0/include
LDFLAGS = -L/usr/local/cuda-13.0/lib64 -lcudart -lnvidia-ml -lpthread

TARGET = busy
OBJS = main.o gpu_kernels.o

all: $(TARGET)

$(TARGET): $(OBJS)
	$(HOST_COMPILER) $(CFLAGS) -o $@ $(OBJS) $(LDFLAGS)

main.o: main.cpp busy.h busy_cuda.h
	$(HOST_COMPILER) $(CFLAGS) -c main.cpp -o $@

gpu_kernels.o: gpu_kernels.cu busy_cuda.h
	$(NVCC) -I./cuda_include -c gpu_kernels.cu -o $@

clean:
	rm -f $(TARGET) $(OBJS)
