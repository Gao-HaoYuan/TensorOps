#include <numeric>
#include <utility>
#include <algorithm>

#include "common.h"
#include "operand.h"

namespace witin {

int64_t OperandLayout::numel() const {
    int64_t numel = 1;
    for (int64_t size : shape_) {
        numel *= size;
    }
    return numel;
}

int64_t OperandLayout::num_output_elements() const {
    int64_t elem = 1;
    for (int dim = 0; dim < ndim(); dim++) {
        if (operands_[0]->stride[dim] != 0 || shape_[dim] == 0)  {
            elem *= shape_[dim];
        }
    }
    return elem;
}

int OperandLayout::num_reduce_dims() const {
    int count = 0;
    for (int dim = 0; dim < ndim(); dim++) {
        if (operands_[0]->stride[dim] == 0) {
            count++;
        }
    }
    return count;
}

void* OperandLayout::data_ptr(int64_t arg) const {
    return operands_[arg]->data;
}

bool OperandLayout::is_dim_reduced(int dim) const {
    for (int arg = 0; arg < num_operand(); arg++) {
        if (operands_[arg]->is_output && operands_[arg]->stride[dim] == 0 && shape_[dim] > 1) {
            return true;
        }
    }
    return false;
}

SmallVector OperandLayout::get_dim_strides(int dim) const {
    auto dims = ndim();
    auto inner_strides = SmallVector();
    for (auto& op : operands_) {
        inner_strides.push_back(dims == 0 ? 0 : op->stride[dim]);
    }
    return inner_strides;
}

std::vector<char*> OperandLayout::get_base_ptrs() const {
    std::vector<char*> ptrs(num_operand());
    std::transform(operands_.begin(), operands_.end(), ptrs.begin(), [](const TensorLayoutPtr op) {
        return static_cast<char*>(op->data);
    });
    return ptrs;
}

bool OperandLayout::is_scalar(int64_t arg) const {
    const auto& stride = operands_[arg]->stride;
    for (int i = 0; i < ndim(); i++) {
        if (stride[i] != 0 && shape_[i] != 1) {
            return false;
        }
    }
    return true;
}

void OperandLayout::compute_shape() {
    all_ops_same_shape_ = true;
    bool has_scalars = false;
    bool has_tensors = false;
    for (auto& op : operands_) {
        auto shape = op->shape;
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

    LOG_DEBUG("compute shape is %s", to_string(shape_).c_str());
}

void OperandLayout::compute_strides() {
    for (auto& op : operands_) {
        SmallVector original_shape = op->shape;

        auto original_stride = op->stride;
        auto offset = ndim() - original_shape.size();
        if (offset > 0)
            op->stride = SmallVector(ndim(), 0);

        for (int i = 0; i < original_shape.size(); i++) {
            if (original_shape[i] == 1 && shape_[offset + i] != 1) {
                op->stride[offset + i] = 0;
            } else {
                op->stride[offset + i] = original_stride[i];
            }
        }
    }
}

bool OperandLayout::fast_set_up(FastSetupType setup_type) {
    if (is_reduction_ || !all_ops_same_shape_) {
        LOG_DEBUG("All the operands' shpae is not same.");
        return false;
    }

    // TODO enable fast handling for reductions, transpose.
    switch (setup_type) {
        case FastSetupType::NONE:
            {
                LOG_DEBUG("fast type is not effect.");
                return false;
            }
        case FastSetupType::CONTIGUOUS:
            {
                LOG_DEBUG("fast type is contiguous.");
                break;
            }
        case FastSetupType::TRANSPOSE:
            {
                LOG_DEBUG("fast type is transpose.");
                break;
            }
        case FastSetupType::CHANNELS_LAST:
            {
                LOG_DEBUG("fast type is channel last.");
                break;
            }
        default:
            WITIN_CHECK(false, "Unsupported fast setup type.");
    }

    // 如果上面的 case 可以通过，那么代表可以直接合并成 1 维计算
    if (ndim() >= 1) {
        shape_[0] = numel();
        shape_.resize(1);
    }

    for (auto& op : operands_ ) {
        op->stride.resize(ndim());
        if (ndim()>0) {
            op->stride[0] = 1;
        }
    }

    return true;
}

void OperandLayout::reorder_dimensions() {
    perm_.resize(ndim());
    if (ndim() == 1) {
        perm_[0] = 0;
        return;
    }

    // initialize perm with n-1, n-2, ..., 1, 0
    for (int i = 0, val = perm_.size() - 1; i < perm_.size(); ++i, --val) {
        perm_[i] = val;
    }

    // 1. 如果是 reduction 且是输出张量，stride 为 0 的维度排在前面.
    // 2. 跳过 broadcast 维度.
    // 3. 按 stride 大小排序：stride 小的维度排在前面.
    // 4. stride 相等时按维度长度降序：维度长度大的排在后面.
    // 5. 若所有 tensor 均无法区分：不交换.
    auto should_swap = [&](size_t dim0, size_t dim1) {
        for (int arg = 0; arg < num_operand(); arg++) {
            // ignore undefined or incorrectly sized tensors
            if (operands_[arg]->stride.empty()) {
                continue;
            }

            int64_t stride0 = operands_[arg]->stride[dim0];
            int64_t stride1 = operands_[arg]->stride[dim1];

            if (is_reduction_ && operands_[arg]->is_output) {
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

void OperandLayout::permute_dimensions(SmallVector perm) {
    WITIN_ASSERT(perm.size() == static_cast<unsigned>(ndim()));

    auto reorder = [perm](SmallVector data) {
        auto res = SmallVector(data.size(), 0);
        for (int i = 0; i < perm.size(); i++) {
            res[i] = data[perm[i]];
        }
        return res;
    };

    shape_ = reorder(shape_);
    for (auto& op : operands_) {
        if (!op->stride.empty()) {
            op->stride = reorder(op->stride);
        }
    }
}

void OperandLayout::coalesce_dimensions() {
    if (ndim() <= 1) {
        return;
    }

    auto can_coalesce = [&](int dim0, int dim1) {
        auto shape0 = shape_[dim0];
        auto shape1 = shape_[dim1];

        if (shape0 == 1 || shape1 == 1) {
            return true;
        }

        for (int i = 0; i < num_operand(); i++) {
            auto& stride = operands_[i]->stride;
            if (shape0 * stride[dim0] != stride[dim1]) {
                return false;
            }
        }
        return true;
    };

    auto replace_stride = [&](int dim0, int dim1) {
        for (int i = 0; i < num_operand(); i++) {
            auto& stride = operands_[i]->stride;
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
    for (int i = 0; i < num_operand(); i++) {
        operands_[i]->stride.resize(ndim());
    }
}

void OperandLayout::setup_layout(FastSetupType setup_type) {
    // compute the broadcasted shape
    compute_shape();

    // try fast setup output tensor, if failed, fallback to normal setup
    if (!fast_set_up(setup_type)) {
        // compute each tensor's stride after broadcasting
        compute_strides();
        // re-order dimensions to improve coalescing
        reorder_dimensions();
        // coalesce adjacent dimensions when possible
        coalesce_dimensions();
    }

    // update operand's shape.
    for (auto op : operands_) {
        op->shape = shape_;
    }

    LOG_DEBUG("shape_ is %s", to_string(shape_).c_str());
    for (int arg = 0; arg < num_operand(); ++arg) {
        LOG_DEBUG("operand[%d] stride is %s", arg, to_string(strides(arg)).c_str());
    }
}

} // namespace witin