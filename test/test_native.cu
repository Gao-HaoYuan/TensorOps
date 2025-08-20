#include <cuda_runtime.h>

#include "common.h"
#include "layout.h"
#include "operand.h"
#include "engine.cuh"
#include "mini_gtest.h"

using namespace witin;

TensorLayoutPtr create_layout(
    void* data,
    const SmallVector& shape,
    const SmallVector& stride,
    const bool is_output = false
) {
    return SetTensorLayout(data, shape, stride, is_output);
}

template<typename T>
struct AddOperator {
    static constexpr int smem_size = 0;
    // TODO: block dim 设置策略
    static constexpr int MaxThreadsPerBlock = 256;
    static constexpr int MinBlocksPerMultiprocessor = 16;

    struct Params {
        TensorLayout ptr_in0;
        TensorLayout ptr_in1;
        TensorLayout ptr_out;

        int numel;

        Params(
            TensorLayoutPtr in0,
            TensorLayoutPtr in1,
            TensorLayoutPtr out,
            int numel_
        ) 
            : ptr_in0(*in0), ptr_in1(*in1), ptr_out(*out), numel(numel_) {}
    };

    static dim3 get_block_shape() { return dim3(MaxThreadsPerBlock, 1, 1); }
    static dim3 get_grid_shape(Params const& params) {
        int x = (params.numel + MaxThreadsPerBlock - 1) / MaxThreadsPerBlock;
        return dim3(x, 1, 1);
    }

    DEVICE void operator()(Params const& params, char* /*smem_buf*/) const {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= params.numel) return;

        T* __restrict__ output_ptr = static_cast<T*>(params.ptr_out.data);
        T* __restrict__ input0_ptr = static_cast<T*>(params.ptr_in0.data);
        T* __restrict__ input1_ptr = static_cast<T*>(params.ptr_in1.data);

        int off_out = params.ptr_out(idx);
        int off_in0 = params.ptr_in0(idx);
        int off_in1 = params.ptr_in1(idx);

        output_ptr[off_out] = input0_ptr[off_in0] + input1_ptr[off_in1];
    }
};

template <typename T>
bool compare(const T* ptr1, const T* ptr2, int numel) {
    bool flag = true;
    const float epsilon = 1e-5;
    for (int i = 0; i < numel; i++) {
        flag &= std::fabs(ptr1[i] - ptr2[i]) < epsilon;
    }
    return flag;
}

TEST(NativeTest, Common) {
    const int height = 1024;
    const int width = 2048;

    const int numel = height * width;
    const int total_size = numel * sizeof(float);

    void *input0, *input1, *output;
    cudaMalloc(&input0, total_size);
    cudaMalloc(&input1, total_size);
    cudaMalloc(&output, total_size);

    float* h_input0 = (float*)malloc(total_size);
    float* h_input1 = (float*)malloc(total_size);
    float* h_output = (float*)malloc(total_size);

    for (int i = 0; i < numel; ++i) {
        h_input0[i] = static_cast<float>(i);
        h_input1[i] = static_cast<float>(i * 2);
    }
    cudaMemcpy(input0, h_input0, total_size, cudaMemcpyHostToDevice);
    cudaMemcpy(input1, h_input1, total_size, cudaMemcpyHostToDevice);

    auto lay_in0 = create_layout(input0, {height, width}, {width, 1});
    auto lay_in1 = create_layout(input1, {height, width}, {width, 1});
    auto lay_out = create_layout(output, {height, width}, {width, 1}, true);

    for (int i = 0; i < numel; ++i) {
        h_output[i] = h_input0[i] + h_input1[i];
    }

    OperandLayout operand(lay_out, lay_in0, lay_in1);
    operand.setup_layout(FastSetupType::CONTIGUOUS);
    std::cout << "lay_in0: " << to_string(*lay_in0) << std::endl;
    std::cout << "lay_in1: " << to_string(*lay_in1) << std::endl;
    std::cout << "lay_out: " << to_string(*lay_out) << std::endl;

    gpu_kernel<AddOperator<float>>(cudaStreamDefault, lay_in0, lay_in1, lay_out, operand.numel());

    float* h_result = (float*)malloc(total_size);
    cudaMemcpy(h_result, output, total_size, cudaMemcpyDeviceToHost);

    cudaFree(input0);
    cudaFree(input1);
    cudaFree(output);

    EXPECT_TRUE(compare(h_output, h_result, numel));
}
