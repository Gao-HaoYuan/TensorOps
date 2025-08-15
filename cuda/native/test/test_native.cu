#include <cuda_runtime.h>

#include "common.h"
#include "layout.h"
#include "operand.h"
#include "engine.cuh"
#include "mini_gtest.h"

using namespace witin;

SharedLayout create_layout(
    void* data,
    const SmallVector& shape,
    const SmallVector& stride,
    const SmallVector& view_offset,
    const ScalarType& dtype,
    const Device& device,
    const MemoryFormat& memory_format,
    const std::string& name = ""
) {
    LayoutOptions options;
    options.set_dtype(dtype);
    options.set_device(device);
    options.set_memory_format(memory_format);

    return MakeLayout(data, shape, stride, view_offset, options, name);
}

template<typename T>
struct AddOperator {
    static constexpr int N = 3;
    static constexpr int MaxThreadsPerBlock = 256;
    static constexpr int MinBlocksPerMultiprocessor = 2;
    static constexpr dim3 block{256, 1, 1};
    using Params = IterParams<N>;

    INLINE_HOST_DEVICE
    void operator()(Params const params, const int64_t* offsets, char*) {
        auto output_ptr = static_cast<T*>(params.ptrs[0]) + offsets[0];
        auto input0_ptr = static_cast<T*>(params.ptrs[1]) + offsets[1];
        auto input1_ptr = static_cast<T*>(params.ptrs[2]) + offsets[2];

        *output_ptr = *input0_ptr + *input1_ptr;
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

    const ScalarType dtype = ScalarType::Float;
    const Device device(DeviceType::CUDA, 0);

    const int numel = height * width;
    const int total_size = numel * elementSize(dtype);

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

    auto lay_in0 = create_layout(input0, {height, width}, {width, 1}, {0, 0}, dtype, device, MemoryFormat::Contiguous, "input0");
    auto lay_in1 = create_layout(input1, {height, width}, {width, 1}, {0, 0}, dtype, device, MemoryFormat::Contiguous, "input1");
    auto lay_out = create_layout(output, {height, width}, {width, 1}, {0, 0}, dtype, device, MemoryFormat::Contiguous, "output");

    OperandLayoutConfig config;
    config.check_all_same_device(true);
    config.check_all_same_dtype(true);
    config.add_output(lay_out);
    config.add_input(lay_in0);
    config.add_input(lay_in1);

    for (int i = 0; i < numel; ++i) {
        h_output[i] = h_input0[i] + h_input1[i];
    }

    OperandLayout operand = config.build();
    gpu_kernel(operand, AddOperator<float>{});

    float* h_result = (float*)malloc(total_size);
    cudaMemcpy(h_result, output, total_size, cudaMemcpyDeviceToHost);

    cudaFree(input0);
    cudaFree(input1);
    cudaFree(output);

    EXPECT_TRUE(compare(h_output, h_result, numel));
}
