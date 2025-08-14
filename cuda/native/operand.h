#pragma once

#include <optional>

#include "layout.h"
#include "meta.h"

namespace witin
{
/**
 * @brief Operand 的信息，如果需要额外信息，再补充，目前只有 stride, dtype, device, data ptr，memory format 信息.
 *        element wise 操作的输出具有统一 shape 并由 OperandLayoutBase 处理
 * 
 */
struct OperandInfo {
    OperandInfo() = default;
    explicit OperandInfo(SharedLayout t) {
        if (t != nullptr) {
            device_ = t->device();
            target_dtype = t->dtype();
            current_dtype = target_dtype;
        }
        set_layout(t);
        validate();
    }

    OperandInfo(const OperandInfo&) = default;
    OperandInfo& operator=(const OperandInfo&) = default;
    OperandInfo(OperandInfo&&) noexcept = default;
    OperandInfo& operator=(OperandInfo&&) noexcept = default;
    ~OperandInfo() = default;

    const SharedLayout layout_ptr() const {
        return layout_;
    }

    const SharedLayout origin_layout_ptr() const {
        return origin_layout_;
    }

    void set_layout(SharedLayout t) {
        layout_ = std::move(t);
    }

    void validate() {
        ASSERT(layout_ != nullptr, "The tensor is undefined.");
    }

    bool is_device_defined() const {
        return device_.has_value();
    }

    bool is_type_defined() const {
        return target_dtype != ScalarType::Undefined;
    }

    LayoutOptions options() const {
        return LayoutOptions(target_dtype).device(device_);
    }

    // Set layout_ to a new value, and store the old layout_ value in origin_layout_.
    void exchange_layout(SharedLayout t);
    void restore_original_layout();

    void* data_ = nullptr;

    // 封装原始的 layout 信息，并且可能会做进一步的优化处理, 比如合并维度等.
    SmallVector stride_;
    std::optional<Device> device_ = std::nullopt;
    ScalarType target_dtype = ScalarType::Undefined;
    ScalarType current_dtype = ScalarType::Undefined;

    bool is_output = false;
    bool is_const = false;
    bool is_read_write = false;
    // NOTE: 未来处理特殊的 output shape.
    bool will_resize = false;

private:
    SharedLayout layout_;
    SharedLayout origin_layout_;
};


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

class OperandLayoutConfig;

/**
 * @brief 主要用来处理 Tensor 的连续性问题，以及计算 Shape、Stride 和维度合并，功能
 *        待扩充. 未来可以以此为基础设计算子合并.
 * 
 */
struct OperandLayoutBase : public LayoutMeta {

    /**
     * @brief 构建算子的输入输出 tensor
     * 
     */
    void build(OperandLayoutConfig&);
    
    int ndim() const { return static_cast<int>(shape_.size()); }
    int nlayouts() const { return static_cast<int>(operands_.size()); }
    int noutputs() const { return num_outputs_; }
    int ninputs() const { return nlayouts() - noutputs(); }
    const SmallVector& shape() const { return shape_; }
    const SmallVector& view_offset() const { return view_offsets_; }

    int64_t numel() const;
    int64_t num_output_elements() const;
    int num_reduce_dims() const;

    void* data_ptr(int64_t arg) const;
    bool is_dim_reduced(int dim) const;
    void remove_operand(int64_t arg);

    SmallVector get_dim_strides(int dim) const;
    std::vector<char*> get_base_ptrs() const;

    const SmallVector& strides(int64_t arg) const { return operands_[arg].stride_; }
    ScalarType dtype(int64_t arg = 0) const { return operands_[arg].current_dtype; }

    ScalarType common_dtype() const {
        ASSERT(common_dtype_ != ScalarType::Undefined, "Queried for invalid common dtype!");
        return common_dtype_;
    }

    ScalarType input_dtype(int64_t arg = 0) const { return operands_[num_outputs_ + arg].current_dtype; }
    Device device(int64_t arg = 0) const { return operands_[arg].device_.value(); }
    DeviceType device_type(int64_t arg = 0) const { return device(arg).type(); }
    int64_t element_size(int64_t arg) const { return static_cast<int64_t>(elementSize(dtype(arg))); }
    
    bool is_scalar(int64_t arg) const;
    bool is_cpu_scalar(int64_t arg) const;

    const SharedLayout layout_ptr(int64_t arg) const { return operands_[arg].layout_ptr(); }

    const SharedLayout output(int64_t arg = 0) const {
        ASSERT(arg < num_outputs_);
        return layout_ptr(arg);
    }

    const SharedLayout input(int64_t arg = 0) const {
        ASSERT(arg >= 0 && arg < nlayouts() - num_outputs_);
        return layout_ptr(num_outputs_ + arg);
    }

    const OperandInfo& operand(int arg = 0) const {
        return operands_[arg];
    }

    OperandInfo& operand(int arg = 0) {
        return operands_[arg];
    }

    void set_output_raw_strided(
        int64_t output_idx,
        SmallVector sizes,
        SmallVector strides,
        LayoutOptions options,
        std::string names
    ) override;

protected:
    void populate_operands(OperandLayoutConfig&);
    void mark_outputs();
    void compute_shape(const OperandLayoutConfig&);
    void compute_strides(const OperandLayoutConfig&);
    void compute_types(const OperandLayoutConfig&);
    bool fast_set_up(const OperandLayoutConfig&);
    FastSetupType compute_fast_setup_type(const OperandLayoutConfig&);
    void reorder_dimensions();
    void permute_dimensions(SmallVector perm);
    void coalesce_dimensions();

protected:
    std::vector<OperandInfo> operands_;

    SmallVector shape_;
    SmallVector perm_;
    // NOTE: Torch will offset the date ptr, the var maybe unused.
    SmallVector view_offsets_;
    std::string name_;

    int num_outputs_ = 0;
    bool is_reduction_ = false;

    bool has_different_input_dtypes = false;
    bool has_different_output_dtypes = false;
    bool has_coalesced_dimensions_ = false;
    bool all_ops_same_shape_ = false;

    // NOTE: 例如为了计算精度，需要提前转换的类型
    ScalarType common_dtype_ = ScalarType::Undefined;
    Device common_device_ = DeviceType::CPU;
};

/**
 * @brief 对 TensorViewSchedulerBase 的封装
 * 
 */
struct OperandLayout final : public OperandLayoutBase {
    OperandLayout() : OperandLayoutBase() {}
    // Slicing is OK, TensorIterator guaranteed NOT to have any fields
    OperandLayout(const OperandLayoutBase& iter) : OperandLayoutBase(iter) {}

    const SharedLayout maybe_get_output(int64_t output_idx) override {
        return output(output_idx);
    }

    void set_output_raw_strided(
        int64_t output_idx,
        SmallVector sizes,
        SmallVector strides,
        LayoutOptions options,
        std::string names
    ) override;
};

/**
 * @brief Operand 的 config 信息.
 * 
 */
class OperandLayoutConfig final {
public:
    friend struct OperandLayoutBase;
    friend struct OperandLayout;

    OperandLayoutConfig() = default;
    DISABLE_COPY_AND_ASSIGN(OperandLayoutConfig);
    OperandLayoutConfig(OperandLayoutConfig&&) = default;
    OperandLayoutConfig& operator=(OperandLayoutConfig&&) = default;
    ~OperandLayoutConfig() = default;

    OperandLayoutConfig& add_output(const SharedLayout output);
    OperandLayoutConfig& add_input(const SharedLayout input);
    OperandLayoutConfig& add_const_input(const SharedLayout input);

    OperandLayoutConfig& check_all_same_dtype(const bool _check_all_same_dtype) {
        check_all_same_dtype_ = _check_all_same_dtype;
        return *this;
    }

    OperandLayoutConfig& check_all_same_device(
        const bool _check_all_same_device) {
        check_all_same_device_ = _check_all_same_device;
        return *this;
    }

    OperandLayoutConfig& is_reduction(const bool _is_reduction) {
        is_reduction_ = _is_reduction;
        return *this;
    }

    OperandLayout build() {
        OperandLayout operand;
        operand.build(*this);
        return operand;
    }

private:
    bool is_layout_const(size_t idx);

    std::vector<SharedLayout> layout_;
    std::vector<size_t> const_layout_indices_;

    int num_outputs_ = 0;
    int num_inputs_ = 0;

    bool check_all_same_dtype_ = true;
    bool check_all_same_device_ = true;

    bool is_reduction_ = false;
};

} // namespace witin
