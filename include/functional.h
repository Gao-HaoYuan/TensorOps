#pragma once

#include <sstream>
#include <string>

#include <torch/extension.h>

#include "macro.h"
#include "layout.h"

using namespace witin;
using torch::Tensor;

inline bool is_transpose(const Tensor& tensor) {
    TORCH_CHECK(tensor.dim() == 2, "tensor must be a 2D tensor");
    if (tensor.is_non_overlapping_and_dense()) { // commen case
        return !tensor.is_contiguous();
    }

    c10::IntArrayRef tensor_strides = tensor.strides();
    c10::IntArrayRef tensor_sizes = tensor.sizes();
    if ((tensor_strides[0] == 1) && (tensor_strides[1] >= std::max<int64_t>(1, tensor_sizes[0]))) {
        return true;
    } else if ((tensor_strides[1] == 1) && (tensor_strides[0] >= std::max<int64_t>(1, tensor_sizes[1]))) {
        return false;
    } else {
        return false;
    }
}

inline std::string get_tensor_info(const Tensor& tensor, const std::string& name = "tensor") {
    std::ostringstream oss;

    oss << "===== " << name << " =====\n";
    oss << "ptr            : " << tensor.data_ptr() << "\n";
    oss << "dim            : " << tensor.dim() << "\n";
    oss << "Type           : " << tensor.toString() << "\n";
    oss << "Shape          : " << tensor.sizes() << "\n";
    oss << "Dtype          : " << tensor.dtype() << "\n";
    oss << "Device         : " << tensor.device() << "\n";
    oss << "Stride         : " << tensor.strides() << "\n";
    oss << "Requires Grad  : " << (tensor.requires_grad() ? "true" : "false") << "\n";
    oss << "Storage Offset : " << tensor.storage_offset() << "\n";
    oss << "Is Leaf        : " << (tensor.is_leaf() ? "true" : "false") << "\n";
    oss << "Is Contiguous  : " << (tensor.is_contiguous() ? "true" : "false") << "\n";

    // 打印前 5 个元素
    auto flat = tensor.flatten();
    oss << "Data (first 5) : [";
    for (int64_t i = 0; i < std::min<int64_t>(5, flat.numel()); ++i) {
        oss << flat[i].item<float>();
        if (i < 4 && i + 1 < flat.numel()) oss << ", ";
    }
    oss << "]\n====================\n";

    return oss.str();
}

// FastSetupType OperandLayout::compute_fast_setup_type() {
//     if (is_reduction_ || !all_ops_same_shape_) {
//         LOG_DEBUG("All the operands' shpae is not same.");
//         return FastSetupType::NONE;
//     }

//     bool is_contiguous = true;
//     bool is_channels_last = true;

//     for (const auto& op : operands_) {
//         if (op->data != nullptr) {
//             is_contiguous &= op->format == MemoryFormat::Contiguous;
//             is_channels_last &= op->format == MemoryFormat::ChannelsLast;
//         }
//     }

//     // TODO: this leads to ambiguous cases (NC11) to be always treated as contiguous
//     if (is_contiguous) {
//         return FastSetupType::CONTIGUOUS;
//     }

//     if (is_channels_last) {
//         return FastSetupType::CHANNELS_LAST;
//     }

//     return FastSetupType::NONE;
// }

inline TensorLayoutPtr get_torch_layout(const Tensor& tensor, bool is_output = false) {
    void* ptr = tensor.data_ptr();

    // get shape
    SmallVector shape;
    for (auto i : tensor.sizes()) {
        shape.push_back(i);
    }

    // get stride
    SmallVector stride;
    for (auto i : tensor.strides()) {
        stride.push_back(i);
    }

    return SetTensorLayout(ptr, shape, stride, is_output);
}