#pragma once

#include <optional>

#include "layout.h"

namespace witin
{
enum class FastSetupType : uint8_t {
    NONE,
    CONTIGUOUS,
    TRANSPOSE,
    CHANNELS_LAST,
    /**
     * @brief torch 接口函数是 is_non_overlapping_and_dense(), 暂时不启用，因为这个限制太多，如果实现
     *        了 reduce, transpose， 可能这个就不需要了，换句话说就是把这个标签的功能细节化.
     * 
     *        torch 里触发下面 case 就不使用这个优化：
     *          1. input 具有不同的 stride
     *          2. ouput 不需要 resize, 具有不同的 stride
     *          3. input 具有相同的 stride，但是和 output stride 不一致，
     *      
     *        这就导致 broadcast, reduce, transpose，部分 permute 也不能使用
     * 
     *        TODO: 细化这个标签的功能.
     */
    NON_OVERLAPPING_DENSE
};

/**
 * @brief 主要用来处理 Tensor 的连续性问题，以及计算 Shape、Stride 和维度合并，功能
 *        待扩充. 未来可以以此为基础设计算子合并.
 * 
 */
struct OperandLayout final {
    DISABLE_COPY_AND_ASSIGN(OperandLayout);

    OperandLayout(OperandLayout&&) = default;
    OperandLayout& operator=(OperandLayout&&) = default;

    OperandLayout() = default;
    ~OperandLayout() = default;

    template<typename... Args>
    OperandLayout(Args&&... args) {
        // fold expression
        (push_if_shared_layout(std::forward<Args>(args)), ...);

        LOG_DEBUG("operand number is %d", num_operand());
    }

    /**
     * @brief shape and stride are processed, and can be simply understood as the 
     *        inverse of the tensor rules.
     *
     * @param setup_type optimize policy.
     */
    void setup_layout(FastSetupType setup_type);

    void is_reduction(const bool _is_reduction) { is_reduction_ = _is_reduction; }
    int ndim() const { return static_cast<int>(shape_.size()); }
    int num_operand() const { return static_cast<int>(operands_.size()); }
    const SmallVector& shape() const { return shape_; }

    int64_t numel() const;
    int64_t num_output_elements() const;
    int num_reduce_dims() const;

    void* data_ptr(int64_t arg) const;
    bool is_dim_reduced(int dim) const;

    SmallVector get_dim_strides(int dim) const;
    std::vector<char*> get_base_ptrs() const;

    const SmallVector& strides(int64_t arg) const { return operands_[arg]->stride; }
    
    bool is_scalar(int64_t arg) const;
    bool is_cpu_scalar(int64_t arg) const;

private:
    void push_if_shared_layout(const TensorLayoutPtr& s) { operands_.push_back(s); }
    void push_if_shared_layout(TensorLayoutPtr&& s) { operands_.push_back(std::move(s)); }
    template<typename T> void push_if_shared_layout(T /* other type */) { }

    void compute_shape();
    void compute_strides();
    bool fast_set_up(FastSetupType setup_type);
    FastSetupType compute_fast_setup_type();
    void reorder_dimensions();
    void permute_dimensions(SmallVector perm);
    void coalesce_dimensions();

    std::vector<TensorLayoutPtr> operands_;
    SmallVector shape_;
    SmallVector perm_;

    bool is_reduction_ = false;
    bool all_ops_same_shape_ = false;
};

} // namespace witin
