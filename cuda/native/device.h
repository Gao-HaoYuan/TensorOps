#pragma once

#include<string>

#include "macro.h"

namespace witin {

enum class DeviceType : int8_t {
    CPU = 0,
    CUDA = 1, // CUDA.
    COMPILE_TIME_MAX_DEVICE_TYPES
};

constexpr int COMPILE_TIME_MAX_DEVICE_TYPES = static_cast<int>(DeviceType::COMPILE_TIME_MAX_DEVICE_TYPES);
static_assert(COMPILE_TIME_MAX_DEVICE_TYPES <= 2, "Now only support CUDA device.");

inline std::string DeviceTypeName(DeviceType d, bool lower_case) {
    switch (d) {
        case DeviceType::CPU:
            return lower_case ? "cpu" : "CPU";
        case DeviceType::CUDA:
            return lower_case ? "cuda" : "CUDA";
        default:
            ASSERT(false, "Unknown device.");
    }
}

inline bool isValidDeviceType(DeviceType d) {
    switch (d) {
        case DeviceType::CPU:
        case DeviceType::CUDA:
            return true;
        default:
            return false;
    }
}

using DeviceIndex = int8_t;

struct Device final {
    using Type = DeviceType;

    Device(DeviceType type, DeviceIndex index = -1) : type_(type), index_(index) {
        validate();
    }

    bool operator==(const Device& other) const noexcept {
        return this->type_ == other.type_ && this->index_ == other.index_;
    }

    bool operator!=(const Device& other) const noexcept {
        return !(*this == other);
    }

    void set_index(DeviceIndex index) {
        index_ = index;
    }

    DeviceType type() const noexcept {
        return type_;
    }

    DeviceIndex index() const noexcept {
        return index_;
    }

    bool has_index() const noexcept {
        return index_ != -1;
    }

    // Return true if the device is of CUDA type.
    bool is_cuda() const noexcept {
        return type_ == DeviceType::CUDA;
    }

    // Return true if the device is of CPU type.
    bool is_cpu() const noexcept {
        return type_ == DeviceType::CPU;
    }

    std::string str() const {
        std::string str = DeviceTypeName(type(), /* lower case */ true);
        if (has_index()) {
            str.push_back(':');
            str.append(std::to_string(index()));
        }
        return str;
    }

 private:
    DeviceType type_;
    DeviceIndex index_ = -1;
    void validate() {
        ASSERT(
            index_ >= -1,
            "Device index must be -1 or non-negative, got ",
            static_cast<int>(index_));
        ASSERT(
            !is_cpu() || index_ <= 0,
            "CPU device index must be -1 or zero, got ",
            static_cast<int>(index_));
    }
};

inline std::ostream& operator<<(std::ostream& stream, const Device& device) {
    stream << device.str();
    return stream;
}

} // namespace witin

namespace std {
template <>
struct hash<witin::DeviceType> {
    std::size_t operator()(witin::DeviceType k) const {
        return std::hash<int>()(static_cast<int>(k));
    }
};

template <>
struct hash<witin::Device> {
    size_t operator()(witin::Device d) const noexcept {
        static_assert(sizeof(witin::DeviceType) == 1, "DeviceType is not 8-bit");
        static_assert(sizeof(witin::DeviceIndex) == 1, "DeviceIndex is not 8-bit");
        
        // 注意：-1 直接转换成无符号数是该类型的最大值，需要先转成 uint8_t
        uint32_t bits = static_cast<uint32_t>(static_cast<uint8_t>(d.type()))
                << 16 |
            static_cast<uint32_t>(static_cast<uint8_t>(d.index()));
        return std::hash<uint32_t>{}(bits);
    }
};
} // namespace std