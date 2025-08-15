#include <numeric>
#include <algorithm>

#include "common.h"
#include "operand.h"

namespace witin {

namespace inner {

static LayoutOptions original_options(const OperandInfo& op) {
    if (op.origin_layout_ptr() != nullptr) {
        return op.origin_layout_ptr()->options();
    } else {
        return op.options();
    }
}

} // namespace inner 

void OperandInfo::exchange_layout(SharedLayout t) {
    ASSERT(origin_layout_ == nullptr);
    origin_layout_ = std::exchange(layout_, std::move(t));
}

void OperandInfo::restore_original_layout() {
    ASSERT(origin_layout_ != nullptr);
    layout_ = std::move(origin_layout_);
}

OperandLayoutConfig& OperandLayoutConfig::add_output(const SharedLayout output) {
    ASSERT(
        num_inputs_ == 0,
        "Keep in mind that you have to add all outputs first before adding any input.");
    layout_.push_back(output);
    num_outputs_++;
    return *this;
}

OperandLayoutConfig& OperandLayoutConfig::add_input(const SharedLayout input) {
    layout_.push_back(input);
    num_inputs_++;
    return *this;
}

OperandLayoutConfig& OperandLayoutConfig::add_const_input(const SharedLayout input) {
    const_layout_indices_.push_back(layout_.size());
    layout_.push_back(input);
    num_inputs_++;
    return *this;  
}

bool OperandLayoutConfig::is_layout_const(size_t idx) {
    return std::find(const_layout_indices_.begin(), const_layout_indices_.end(), idx) != const_layout_indices_.end();
}

int64_t OperandLayoutBase::numel() const {
    int64_t numel = 1;
    for (int64_t size : shape_) {
        numel *= size;
    }
    return numel;
}

int64_t OperandLayoutBase::num_output_elements() const {
    int64_t elem = 1;
    for (int dim = 0; dim < ndim(); dim++) {
        if (operands_[0].stride_[dim] != 0 || shape_[dim] == 0)  {
            elem *= shape_[dim];
        }
    }
    return elem;
}

int OperandLayoutBase::num_reduce_dims() const {
    int count = 0;
    for (int dim = 0; dim < ndim(); dim++) {
        if (operands_[0].stride_[dim] == 0) {
            count++;
        }
    }
    return count;
}

void* OperandLayoutBase::data_ptr(int64_t arg) const {
    return operands_[arg].data_;
}

void OperandLayoutBase::remove_operand(int64_t arg) {
    operands_.erase(operands_.begin() + arg);
}

bool OperandLayoutBase::is_dim_reduced(int dim) const {
    for (auto& op : operands_) {
        if (op.is_output && op.stride_[dim] == 0 && shape_[dim] > 1) {
            return true;
        }
    }
    return false;
}

SmallVector OperandLayoutBase::get_dim_strides(int dim) const {
    auto dims = ndim();
    auto inner_strides = SmallVector();
    for (auto& op : operands_) {
        inner_strides.push_back(dims == 0 ? 0 : op.stride_[dim]);
    }
    return inner_strides;
}

std::vector<char*> OperandLayoutBase::get_base_ptrs() const {
    std::vector<char*> ptrs(nlayouts());
    std::transform(operands_.begin(), operands_.end(), ptrs.begin(), [](const OperandInfo& op) {
        return static_cast<char*>(op.data_);
    });
    return ptrs;
}

bool OperandLayoutBase::is_scalar(int64_t arg) const {
    const auto& stride = operands_[arg].stride_;
    for (int i = 0; i < ndim(); i++) {
        if (stride[i] != 0 && shape_[i] != 1) {
            return false;
        }
    }
    return true;
}

bool OperandLayoutBase::is_cpu_scalar(int64_t arg) const {
    return is_scalar(arg) && device(arg).is_cpu();
}

void OperandLayoutBase::set_output_raw_strided(
    int64_t output_idx,
    SmallVector sizes,
    SmallVector strides,
    LayoutOptions options,
    std::string names
) {
    auto& op = operands_[output_idx];
    // OperandLayoutBase 不会被直接调用，后续在写派生类，下面函数需要派生类自实现
    const auto t = maybe_get_output(output_idx);
    ASSERT(t != nullptr);

    if (op.layout_ptr() == nullptr) {
        op.set_layout(t);
        ASSERT(op.target_dtype == t->dtype());
    } else if (op.will_resize) {
        // 本意这个调度器只处理 Layout, 但是如果 output 的形状过于奇怪，或者
        // 是被指定的内存，这里需要额外特殊处理, 包括申请临时内存.
    }

    ASSERT(op.layout_ptr() == t || op.current_dtype == op.layout_ptr()->dtype());
    op.current_dtype = op.layout_ptr()->dtype();
}

void OperandLayoutBase::populate_operands(OperandLayoutConfig& config) {
    for (int idx = 0; idx < config.layout_.size(); idx++) {
        auto& layout = config.layout_[idx];
        operands_.emplace_back(std::move(layout));
        operands_[idx].is_const = config.is_layout_const(idx);
    }
    num_outputs_ = config.num_outputs_;
}

void OperandLayoutBase::mark_outputs() {
    for (int i = 0; i < num_outputs_; i++) {
        operands_[i].is_output = true;
        const auto output = layout_ptr(i);
        if (output == nullptr) continue;

        // check if output is also an input
        for (int arg = num_outputs_; arg < nlayouts(); arg++) {
            const auto input = layout_ptr(arg);
            if (output == input) {
                operands_[i].is_read_write = true;
            }
        }
    }
}

void OperandLayoutBase::compute_shape(const OperandLayoutConfig& config) {
    all_ops_same_shape_ = true;
    bool has_scalars = false;
    bool has_tensors = false;
    for (auto& op : operands_) {
        if (op.layout_ptr() == nullptr) continue;

        if (op.is_output) continue;

        auto shape = op.layout_ptr()->shape();
        if (shape.empty()) {
            has_scalars = true;
        } else {
            has_tensors = true;
        }

        if (has_scalars && has_tensors) {
            all_ops_same_shape_ = false;
        }

        if (shape_.empty()) {
            shape_ = shape;
        } else if (shape != shape_) {
            all_ops_same_shape_ = false;
            shape_ = infer_layout_shape(shape_, shape);
        }
    }
}

void OperandLayoutBase::compute_strides(const OperandLayoutConfig& config) {
    for (auto& op : operands_) {
        if (op.layout_ptr() && !op.will_resize) {
            SmallVector original_shape = op.layout_ptr()->shape();

            auto original_stride = op.layout_ptr()->stride();
            auto offset = ndim() - original_shape.size();
            if (offset > 0)
                op.stride_.resize(ndim(), 0);
            else
                op.stride_.resize(ndim());

            for (int i = 0; i < original_shape.size(); i++) {
                if (original_shape[i] == 1 && shape_[offset + i] !=1) {
                    op.stride_[offset + i] = 0;
                } else {
                    op.stride_[offset + i] = original_stride[i];
                }
            }
        }
    }
}

void OperandLayoutBase::compute_types(const OperandLayoutConfig& config) {
    Device common_device = DeviceType::CPU;
    common_dtype_ = ScalarType::Undefined;

    ScalarType output_dtype = ScalarType::Undefined;

    for (auto& op : operands_) {
        ASSERT(op.target_dtype == op.current_dtype);

        if (common_device == DeviceType::CPU && !op.layout_ptr()->device().is_cpu()) {
            common_device = op.layout_ptr()->device();
        }

        if (!op.is_output) {
            // Determines if there are varying input dtypes
            // NOTE: the common dtype is set to the first defined input dtype observed
            if (op.target_dtype != common_dtype_) {
                if (common_dtype_ == ScalarType::Undefined) {
                    common_dtype_ = op.target_dtype;
                } else {
                    has_different_input_dtypes = true;
                }
            }
        } else {  // op.is_output
            // Determines if there are varying output dtypes
            // NOTE: the output dtype is set to the first defined output dtype observed
            if (op.target_dtype != output_dtype) {
                if (output_dtype == ScalarType::Undefined) {
                    output_dtype = op.target_dtype;
                } else {
                    has_different_output_dtypes = true;
                }
            }
        }
    }

    if (config.check_all_same_dtype_ && (has_different_input_dtypes || has_different_output_dtypes ||
                            (common_dtype_ != output_dtype && output_dtype != ScalarType::Undefined))) {
        // Throws an informative error message
        for (auto& op : operands_) {
            if (op.layout_ptr() == nullptr) {
                continue;
            }

            ASSERT(op.target_dtype == common_dtype_,
                    "Found dtype ", op.target_dtype, " but expected ", common_dtype_);
        }
    }

    common_device_ = common_device;
    for (auto& op : operands_) {
        bool is_type_defined = op.is_type_defined();
        bool is_device_defined = op.is_device_defined();

        if (!is_type_defined) {
            op.target_dtype = common_dtype_;
        }
        if (!is_device_defined) {
            op.device_ = common_device;
        }

        if (!is_type_defined && !is_device_defined) {
            continue;
        }

        if (config.check_all_same_device_) {
            if (op.device_ != common_device) {
                ASSERT(false, "Found at least two devices, ", common_device, " and ", op.device_, "!");
            }
        }
    }
}

FastSetupType OperandLayoutBase::compute_fast_setup_type(const OperandLayoutConfig& config) {
    if (is_reduction_ || !all_ops_same_shape_) {
        return FastSetupType::NONE;
    }

    bool is_contiguous = true;
    bool is_channels_last = true;

    for (const auto& op : operands_) {
        if (op.layout_ptr() != nullptr && !op.will_resize) {
            is_contiguous &= op.layout_ptr()->memory_format() == MemoryFormat::Contiguous;
            is_channels_last &= op.layout_ptr()->memory_format() == MemoryFormat::ChannelsLast;
        }
    }

    // TODO: this leads to ambiguous cases (NC11) to be always treated as contiguous
    if (is_contiguous) {
        return FastSetupType::CONTIGUOUS;
    }

    if (is_channels_last) {
        return FastSetupType::CHANNELS_LAST;
    }

    return FastSetupType::NONE;
}

bool OperandLayoutBase::fast_set_up(const OperandLayoutConfig& config) {
    // TODO enable fast handling for reductions, transpose.
    FastSetupType setup_type = compute_fast_setup_type(config);
    if (setup_type == FastSetupType::NONE) {
        return false;
    }

    const char palcehold[] = "Outpt is fast set up. Maybe used in the future.";

    switch (setup_type) {
        case FastSetupType::CONTIGUOUS:
            {
                for (int i = 0; i < num_outputs_; i++) {
                    auto& op = operands_[i];
                    if (op.layout_ptr() == nullptr) {
                        ASSERT(op.is_type_defined(), "no type for operand", i);
                    }
                    set_output_raw_strided(i, shape_, {}, inner::original_options(op).memory_format(MemoryFormat::Contiguous), palcehold);
                }
                break;
            }
        case FastSetupType::TRANSPOSE:
            {

            }
        case FastSetupType::CHANNELS_LAST:
            {
                for (int i = 0; i < num_outputs_; i++) {
                    auto& op = operands_[i];
                    if (op.layout_ptr() == nullptr) {
                        ASSERT(op.is_type_defined(), "no type for operand", i);
                    }
                    set_output_raw_strided(i, shape_, {}, inner::original_options(op).memory_format(MemoryFormat::ChannelsLast), palcehold);
                }
                break;
            }
        default:
            ASSERT(false, "Unsupported fast setup type", std::to_string((int)setup_type));
    }

    // 如果上面的 case 可以通过，那么代表可以直接合并成 1 维计算
    if (ndim() > 1){
        has_coalesced_dimensions_ = true;
    }

    if (ndim() >= 1) {
        shape_[0] = numel();
        shape_.resize(1);
    }

    for (auto& op : operands_ ) {
        op.stride_.resize(ndim());
        if (ndim()>0) {
            op.stride_[0] = 1;
        }
    }

    return true;
}

void OperandLayoutBase::reorder_dimensions() {
    perm_.resize(ndim());
    if (ndim() == 1) {
        perm_[0] = 0;
        return;
    }

    // initialize perm with n-1, n-2, ..., 1, 0
    std::iota(perm_.rbegin(), perm_.rend(), 0);

    // 1. 如果是 reduction 且是输出张量，stride 为 0 的维度排在前面.
    // 2. 跳过 broadcast 维度.
    // 3. 按 stride 大小排序：stride 小的维度排在前面.
    // 4. stride 相等时按维度长度降序：维度长度大的排在后面.
    // 5. 若所有 tensor 均无法区分：不交换.
    auto should_swap = [&](size_t dim0, size_t dim1) {
        for (int arg = 0; arg < nlayouts(); arg++) {
            // ignore undefined or incorrectly sized tensors
            if (operands_[arg].stride_.empty() || operands_[arg].will_resize) {
                continue;
            }

            int64_t stride0 = operands_[arg].stride_[dim0];
            int64_t stride1 = operands_[arg].stride_[dim1];

            if (is_reduction_ && operands_[arg].is_output) {
                if ((stride0 == 0) != (stride1 == 0)) {
                    return stride1 == 0 ? 1 : -1;
                }
            }

            if (stride0 == 0 || stride1 == 0) {
                continue;
            } else if (stride0 < stride1) {
                return -1;
            } else  if (stride0 > stride1) {
                return 1;
            } else {
                auto t_dim0 = shape_[dim0];
                auto t_dim1 = shape_[dim1];
                if (t_dim0 > t_dim1) {
                    return 1;
                }
            }
        }
        return 0;
    };


    for (int i = 1; i < ndim(); i++) {
        int dim1 = i;
        for (int dim0 = i - 1; dim0 >= 0; dim0--) {
            int comparison = should_swap(perm_[dim0], perm_[dim1]);
            if (comparison > 0) {
                std::swap(perm_[dim0], perm_[dim1]);
                dim1 = dim0;
            } else if (comparison < 0) {
                break;
            }
        }
    }

    permute_dimensions(perm_);
}

void OperandLayoutBase::permute_dimensions(SmallVector perm) {
    ASSERT(perm.size() == static_cast<unsigned>(ndim()));

    auto reorder = [perm](SmallVector data) {
        auto res = SmallVector(data.size(), 0);
        for (int i = 0; i < perm.size(); i++) {
            res[i] = data[perm[i]];
        }
        return res;
    };

    shape_ = reorder(shape_);
    for (auto& op : operands_) {
        if (!op.stride_.empty()) {
            op.stride_ = reorder(op.stride_);
        }
    }
}

void OperandLayoutBase::coalesce_dimensions() {
    if (ndim() <= 1) {
        return;
    }

    auto can_coalesce = [&](int dim0, int dim1) {
        auto shape0 = shape_[dim0];
        auto shape1 = shape_[dim1];

        if (shape0 == 1 || shape1 == 1) {
            return true;
        }

        for (int i = 0; i < nlayouts(); i++) {
            auto& stride = operands_[i].stride_;
            if (shape0 * stride[dim0] != stride[dim1]) {
                return false;
            }
        }
        return true;
    };

    auto replace_stride = [&](int dim0, int dim1) {
        for (int i = 0; i < nlayouts(); i++) {
            auto& stride = operands_[i].stride_;
            stride[dim0] = stride[dim1];
        }
    };

    int prev_dim = 0;
    for (int dim = 1; dim < ndim(); dim++) {
        if (can_coalesce(prev_dim, dim)) {
            if (shape_[prev_dim] == 1) {
                replace_stride(prev_dim, dim);
            }
            shape_[prev_dim] *= shape_[dim];
        } else {
            prev_dim++;
            if (prev_dim != dim) {
                replace_stride(prev_dim, dim);
                shape_[prev_dim] = shape_[dim];
            }
        }
    }

    shape_.resize(prev_dim + 1);
    for (int i = 0; i < nlayouts(); i++) {
        operands_[i].stride_.resize(ndim());
    }
    has_coalesced_dimensions_ = true;
}

void OperandLayoutBase::build(OperandLayoutConfig& config) {
    is_reduction_ = config.is_reduction_;

    // fill in operands_ based on configuration
    populate_operands(config);
    // set is_output and is_read_write flags on appropriate tensors
    mark_outputs();
    // compute the broadcasted shape
    compute_shape(config);
    // compute the result dtype and device
    compute_types(config);

    // try fast setup output tensor, if failed, fallback to normal setup
    if (!fast_set_up(config)) {
        // compute each tensor's stride after broadcasting
        compute_strides(config);
        // re-order dimensions to improve coalescing
        reorder_dimensions();
        // coalesce adjacent dimensions when possible
        coalesce_dimensions();
    }

    for (auto& op : operands_) {
        ASSERT(op.layout_ptr() != nullptr);
        op.data_ = op.layout_ptr()->data();
    }

    int64_t ndim_offsets = (ndim() ? ndim() : 1);
    view_offsets_ = SmallVector(ndim_offsets, 0);
}

// TODO: name 的作用还没有想好，可以用来生成 hash, 未来用来辅助算子合并.
void OperandLayout::set_output_raw_strided(
    int64_t output_idx,
    SmallVector sizes,
    SmallVector strides,
    LayoutOptions options,
    std::string names
) {
    auto& op = operands_[output_idx];
    ASSERT(output_idx < num_outputs_);

    if (op.layout_ptr() == nullptr) {
        // TODO: 这里暂时不处理
        ASSERT(false, "OperandLayout::set_output_raw_strided not implemented");
    } else if (op.will_resize) {
        // 增加一些 Resize 逻辑
    }
}

} // namespace witin