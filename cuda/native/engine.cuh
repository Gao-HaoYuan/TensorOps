#pragma once

#include <vector>
#include <iostream>

#include <cuda_runtime.h>

#include "macro.h"
#include "operand.h"

namespace witin {

// NOTE: 目前维度受限
constexpr int MAX_NDIMS = 8;

template <int N>
struct IterParams {
    void* ptrs[N];
    int64_t shape[MAX_NDIMS];
    int64_t stride[N * MAX_NDIMS];
    int numel;
    int ndim;
};

template <int NARGS>
INLINE_HOST_DEVICE
void compute_offsets(int64_t idx, const int64_t* sizes, const int64_t* strides, int64_t* offsets, int ndims) {
    #pragma unroll
    for (int t = 0; t < NARGS; ++t) {
        offsets[t] = 0;
    }

    #pragma unroll
    for (int dim = 0; dim < ndims; ++dim) { 
        int64_t div = idx / sizes[dim];
        int64_t mod = idx % sizes[dim];
        idx = div;

        #pragma unroll
        for (int arg = 0; arg < NARGS; ++arg) {
            offsets[arg] += mod * strides[arg * ndims + dim];
        }
    }
}

template <typename Operator>
__global__ static
#ifdef __CUDACC__
__launch_bounds__(Operator::MaxThreadsPerBlock, Operator::MinBlocksPerMultiprocessor)
#endif // __CUDACC__
void device_kernel(GRID_CONSTANT typename Operator::Params const params) {
    // Dynamic shared memory base pointer
    extern __shared__ char smem[];
    Operator op;

    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= params.numel) return;

    int64_t offsets[Operator::N];
    compute_offsets<Operator::N>(idx, params.shape, params.stride, offsets, params.ndim);

    op(params, offsets, smem);
}

template <typename Kernel, typename Params>
void kernel_launch(
    dim3 const grid_dims,
    dim3 const block_dims,
    size_t const smem_size,
    cudaStream_t cuda_stream,
    const Params &kernel_params,
    bool launch_with_pdl = false
) {
    if (!launch_with_pdl) {
        device_kernel<Kernel><<<grid_dims, block_dims, smem_size, cuda_stream>>>(kernel_params);
    } else {
#if ((__CUDACC_VER_MAJOR__ >= 12) || ((__CUDACC_VER_MAJOR__ == 11) && (__CUDACC_VER_MINOR__ >= 8)))
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 900)
        ASSERT(false, "Programmatic dependent launch (PDL) is only supported for SM90.");
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

        cudaError_t launch_result = cudaLaunchKernelEx(&config, &device_kernel<Kernel>, kernel_params);
        if (cudaSuccess != launch_result) {
            ASSERT(false, "kernel_launch: cudaLaunchKernelEx failed with error: ", cudaGetErrorString(launch_result));
        }
#else
        ASSERT(false, "Programmatic dependent launch (PDL) is only supported starting CUDA 11.8.");
#endif
    }

    cudaError_t result = cudaGetLastError();
    ASSERT(cudaSuccess == result, "Kernel launch failed. Reason: ", cudaGetErrorString(result));
}

template <typename func_t>
void gpu_kernel(
    OperandLayoutBase& iter, 
    const func_t& f, 
    cudaStream_t stream = cudaStreamDefault
) {
    const int operands = iter.nlayouts();
    ASSERT(operands == func_t::N, "The number of operands is error. operands=%d, func_t::N=%d\n", operands, func_t::N);
    for (int arg = 0; arg < operands; arg++) {
        ASSERT(iter.device(arg).is_cuda(), "args:", arg, ": is not a CUDA Variable.");
    }

    if (iter.numel() == 0) return;

    // packing params
    using ParamsT = typename func_t::Params;
    ParamsT params;
    params.numel = iter.numel();
    params.ndim = iter.ndim();
    ASSERT(params.ndim <= MAX_NDIMS, "Large-dimension tensors are not supported.");

    const SmallVector& shape = iter.shape();
    for (int i = 0; i < params.ndim; i++) {
        params.shape[i] = shape[i];
    }

    for (int i = 0; i < operands; i++) {
        params.ptrs[i] = iter.data_ptr(i);

        const SmallVector& stride = iter.strides(i);
        for (int dim = 0; dim < params.ndim; dim++) {
            params.stride[i * MAX_NDIMS + dim] = stride[dim];
        }
    }

    const dim3 block = func_t::block;
    const dim3 grid((params.numel + block.x - 1) / block.x);
    // NOTE: element-wise 操作不需要共享内存
    // const size_t smem_size = func_t::smem_size;

    kernel_launch<func_t>(grid, block, 0, stream, params);
}

} // namespace witin
