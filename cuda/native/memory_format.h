#pragma once

#include <iostream>

#include "macro.h"

namespace witin {

enum class MemoryFormat : int8_t {
    Contiguous,
    Transpose,
    ChannelsLast,
    NumOptions
};

inline std::ostream& operator<<(std::ostream& stream, MemoryFormat memory_format) {
    switch (memory_format) {
        case MemoryFormat::Contiguous:
            return stream << "Contiguous";
        case MemoryFormat::Transpose:
            return stream << "Transpose";
        case MemoryFormat::ChannelsLast:
            return stream << "ChannelsLast";
        default:
            ASSERT(false, "Unknown memory format ", memory_format);
    }
}

} // namespace witin