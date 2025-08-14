#pragma once

#include <iostream>
#include <string>
#include <algorithm>

#include "macro.h"
#include "layout.h"

namespace witin {

inline SmallVector contiguous_strides(const SmallVector sizes) {
    const int dims = sizes.size();
    SmallVector strides(dims, 1);
    for (auto i = dims - 2; i >= 0; --i) {
        // Strides can't be 0 even if sizes are 0.
        strides[i] = strides[i + 1] * std::max(sizes[i + 1], int64_t(1));
    }
    return strides;
}

inline std::ostream& operator<<(std::ostream& out, const SmallVector v) {
    out << "[";
    for (size_t i = 0; i < v.size(); ++i) {
        if (i > 0) out << ", ";
        out << v[i];
    }
    out << "]";
    return out;
}

inline std::ostream& operator<<(std::ostream& os, const Layout& layout) {
    os << layout.name() << " Layout[";
    os << "shape=" << layout.shape() << ", ";
    os << "stride=" << layout.stride() << ", ";
    os << layout.options() << "]";
    return os;
}


inline SmallVector infer_layout_shape(SmallVector a, SmallVector b) {
    // Use ptrdiff_t to ensure signed comparison.
    auto dimsA = static_cast<ptrdiff_t>(a.size());
    auto dimsB = static_cast<ptrdiff_t>(b.size());
    auto ndim = dimsA > dimsB ? dimsA : dimsB;
    SmallVector expandedSizes(ndim);

    for (ptrdiff_t i = ndim - 1; i >= 0; --i) {
        ptrdiff_t offset = ndim - 1 - i;
        ptrdiff_t dimA = dimsA - 1 - offset;
        ptrdiff_t dimB = dimsB - 1 - offset;
        auto sizeA = (dimA >= 0) ? a[dimA] : 1;
        auto sizeB = (dimB >= 0) ? b[dimB] : 1;

        ASSERT(
            sizeA == sizeB || sizeA == 1 || sizeB == 1,
            "The size of shape a (", sizeA,
            ") must match the size of shape b (", sizeB,
            ") at non-singleton dimension ", i);

        // 1s map to the other size (even 0).
        expandedSizes[i] = sizeA == 1 ? sizeB : sizeA;
    }

    return expandedSizes;
}

} // namespace witin