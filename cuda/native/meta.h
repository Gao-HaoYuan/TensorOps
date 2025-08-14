#pragma once

#include <string>

#include "macro.h"
#include "options.h"
#include "common.h"

namespace witin {

/**
 * @brief 模仿 torch 写的 meta 类，后期可根据需要更新，主要目的是用来
 *        计算 Output 的 shape/stride.
 * 
 */
struct LayoutMeta {
    LayoutMeta() = default;
    LayoutMeta(const LayoutMeta&) = default;
    LayoutMeta& operator=(const LayoutMeta&) = default;
    LayoutMeta(LayoutMeta&&) noexcept = default;
    LayoutMeta& operator=(LayoutMeta&&) noexcept = default;

    virtual const SharedLayout maybe_get_output(int64_t output_idx) = 0;

    virtual void set_output_strided(
        int64_t output_idx [[maybe_unused]],
        SmallVector sizes [[maybe_unused]],
        SmallVector strides [[maybe_unused]],
        LayoutOptions options [[maybe_unused]],
        std::string name [[maybe_unused]] = {}
    ) {
        ASSERT(false, "set_output_strided not implemented.");
    }

    // 未来会对一些特殊的形状做处理.
    virtual void set_output_raw_strided(
        int64_t output_idx [[maybe_unused]],
        SmallVector sizes [[maybe_unused]],
        SmallVector strides_hint [[maybe_unused]],
        LayoutOptions options [[maybe_unused]],
        std::string name [[maybe_unused]] = {}
    ) {
        ASSERT(false, "set_output_strided not implemented.");
    }

    void set_output_contiguous(
        int64_t output_idx,
        SmallVector sizes,
        LayoutOptions options [[maybe_unused]],
        std::string names = {}
    ) {
        auto strides = contiguous_strides(sizes);
        set_output_strided(output_idx, sizes, strides, options, names);
    }

    const SharedLayout maybe_get_output() {
        return maybe_get_output(0);
    }
    virtual ~LayoutMeta() = default;
};

} // namespace witin