#pragma once

#include <vector>
#include <iostream>

#include <cuda_runtime.h>

#include "macro.h"
#include "common.h"
#include "operand.h"

namespace witin {

template <typename Operator>
__global__ static
void device_kernel(
    GRID_CONSTANT typename Operator::Params const params
) {
    //  Dynamic shared memory base pointer.
    // extern __shared__ char smem[];

    Operator op;
    // Element-wise Operator maybe not need shared memory.
    op(params, nullptr);
}

template <typename Kernel, typename Params>
void kernel_launch(
    dim3 const grid_dims,
    dim3 const block_dims,
    size_t const smem_size,
    cudaStream_t cuda_stream,
    const Params &params,
    bool launch_with_pdl
) {
    if (!launch_with_pdl) {
        device_kernel<Kernel><<<grid_dims, block_dims, smem_size, cuda_stream>>>(params);
    } else {
#if ((__CUDACC_VER_MAJOR__ >= 12) || ((__CUDACC_VER_MAJOR__ == 11) && (__CUDACC_VER_MINOR__ >= 8)))
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 900)
        WITIN_CHECK(false, "Programmatic dependent launch (PDL) is only supported for SM90.");
#endif
        cudaLaunchConfig_t config;
        cudaLaunchAttribute attrs[1];

        config.gridDim = grid_dims;
        config.blockDim = block_dims;
        config.dynamicSmemBytes = smem_size;
        config.stream = cuda_stream;

        config.attrs = attrs;
        attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
        attrs[0].val.programmaticStreamSerializationAllowed = 1;
        config.numAttrs = 1;

        cudaError_t launch_result = cudaLaunchKernelEx(&config, &device_kernel<Kernel>, params);
        if (cudaSuccess != launch_result) {
            WITIN_CHECK(false, "kernel_launch: cudaLaunchKernelEx failed with error: ", cudaGetErrorString(launch_result));
        }
#else
        WITIN_CHECK(false, "Programmatic dependent launch (PDL) is only supported starting CUDA 11.8.");
#endif
    }

    cudaError_t result = cudaGetLastError();
    WITIN_CHECK(cudaSuccess == result, "Kernel launch failed. Reason: ", cudaGetErrorString(result));
}

template <typename Operator, typename ... Args>
void gpu_kernel(
    cudaStream_t stream,
    Args&&... args
) {
    // initialize params
    using ParamsT = typename Operator::Params;
    ParamsT params(std::forward<Args>(args)...);
    
    const dim3 block = Operator::get_block_shape();
    const dim3 grid = Operator::get_grid_shape(params);
    // NOTE: maybe element-wise 操作不需要共享内存
    const size_t smem_size = Operator::smem_size;

    // NOTE: 目前硬编码这一部分
    const bool launch_with_pdl = false;

    kernel_launch<Operator>(grid, block, smem_size, stream, params, launch_with_pdl);
}

} // namespace witin
