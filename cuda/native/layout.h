#pragma once

#include <memory>
#include <vector>

#include "macro.h"
#include "options.h"

namespace witin {

struct Layout {
    Layout(
        void* d,
        const SmallVector& shape,
        const SmallVector& stride,
        const LayoutOptions& options,
        const std::string& name = ""
    ) : 
        data_(d),
        shape_(std::move(shape)),
        stride_(std::move(stride)),
        name_(std::move(name)) 
    {
        if (!options.has_dtype() || !options.has_device() || !options.has_memory_format()) {
            ASSERT(false, "LayoutOptions is not completed. ");
        }
        options_.set_dtype(options.dtype());
        options_.set_device(options.device());
        options_.set_memory_format(options.memory_format());
    }

    DISABLE_COPY_AND_ASSIGN(Layout);

    void* data() const noexcept { return data_; }
    const SmallVector& shape() const noexcept { return shape_; }
    const SmallVector& stride() const noexcept { return stride_; }
    const LayoutOptions& options() const noexcept { return options_; }
    const ScalarType& dtype() const noexcept { return options_.dtype(); }
    const Device& device() const noexcept { return options_.device(); }
    const MemoryFormat& memory_format() const noexcept { return options_.memory_format(); }

    const std::string& name() const noexcept { return name_; }

private:
    void* data_;

    SmallVector shape_;
    SmallVector stride_;
    LayoutOptions options_;
    std::string name_;

private:
    Layout() = delete;
};

#define SharedLayout std::shared_ptr<Layout>
#define MakeLayout std::make_shared<Layout>
}; // end of namespace witin