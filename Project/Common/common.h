#ifndef _COMMON_H
#define _COMMON_H

#include <stdio.h>
#include <sys/time.h>

#define CHECK(call)                                                            \
{                                                                              \
    const cudaError_t error = call;                                            \
    if (error != cudaSuccess)                                                  \
    {                                                                          \
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);                 \
        fprintf(stderr, "code: %d, reason: %s\n", error,                       \
                cudaGetErrorString(error));                                    \
    }                                                                          \
}

inline double seconds() {
    struct timeval tp;
    struct timezone tzp;
    int i = gettimeofday(&tp, &tzp);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}

inline void device_name() {
    // set up device
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("device %d: %s\n", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));
}

inline void device_feat() {
  int dev = 0; // consider only one device
  int driverVersion = 0, runtimeVersion = 0;
  cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, dev);
	cudaDriverGetVersion(&driverVersion);
	cudaRuntimeGetVersion(&runtimeVersion);
  int maxWarps = deviceProp.maxThreadsPerMultiProcessor / deviceProp.warpSize;

  printf("\n\nDevice %d: \"%s\"\n", dev, deviceProp.name);
  printf("------------------------------------------------------------\n");
  printf("  CUDA Capability Major/Minor version number:    %d.%d\n", deviceProp.major, deviceProp.minor);
  printf("  Total amount of global memory:                 %.2f MB\n", (float) deviceProp.totalGlobalMem / 1048576.0f);
  printf("  Total amount of shared memory per block        %.2f KB\n", (float)deviceProp.sharedMemPerBlock / 1024.0f);
  printf("  Total shared memory per multiprocessor:        %.2f KB\n", (float)deviceProp.sharedMemPerMultiprocessor / 1024.0f);
  printf("  Total number of registers available per block  %d\n", deviceProp.regsPerBlock);
  printf("  Maximum number of threads per multiprocessor   %d\n", deviceProp.maxThreadsPerMultiProcessor);
  printf("  Maximum number of warps per multiprocessor     %d\n", maxWarps);
  printf("------------------------------------------------------------\n\n");
}

typedef unsigned long ulong;
typedef unsigned int uint;

#endif // _COMMON_H
