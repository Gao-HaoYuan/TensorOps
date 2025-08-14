#pragma once

#include <cstdint>
#include <vector>
#include <iostream>

#include "logger.h"


#define ASSERT(cond, ...)                                               \
    do {                                                                \
        if (!(cond)) {                                                  \
            LOG_ERROR("Assert: " #cond __VA_OPT__(", " ) __VA_ARGS__);  \
        }                               \
    } while(0)


#define DISABLE_COPY_AND_ASSIGN(classname)          \
  classname(const classname&) = delete;             \
  classname& operator=(const classname&) = delete


#ifdef __CUDACC__
#define HOST_DEVICE __host__ __device__
#define INLINE_HOST_DEVICE __forceinline__ __host__ __device__
#else
#define HOST_DEVICE
#define INLINE_HOST_DEVICE inline
#endif


using SmallVector = std::vector<int64_t>;


#if ((__CUDACC_VER_MAJOR__ >= 12) || ((__CUDACC_VER_MAJOR__ == 11) && (__CUDACC_VER_MINOR__ >= 7))) \
        && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 700)
#define GRID_CONSTANT __grid_constant__
#else
#define GRID_CONSTANT
#endif