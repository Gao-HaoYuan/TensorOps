#pragma once

#include <cuda_runtime.h>

template <typename F, typename... Args>
__global__ void device_invoke(F func, Args... args) {
    func(args...);
}

#define LAUNCH_DEVICE_TEST(GRID, BLOCK, FUNC, ...) \
    device_invoke<<<GRID, BLOCK>>>(FUNC, __VA_ARGS__); cudaDeviceSynchronize();
