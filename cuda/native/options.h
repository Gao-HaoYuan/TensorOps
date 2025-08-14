#pragma once

#include <optional>

#include "dtype.h"
#include "device.h"
#include "memory_format.h"

namespace witin {

struct LayoutOptions {
    LayoutOptions() :
        has_device_(false),
        has_dtype_(false),
        has_memory_format_(false) {}

    LayoutOptions(ScalarType dtype) : LayoutOptions() {
        this->set_dtype(dtype);
    }

    LayoutOptions(MemoryFormat memory_format) : LayoutOptions() {
        this->set_memory_format(memory_format);
    }

    template <typename T, typename = std::enable_if_t<std::is_same_v<std::decay_t<T>, Device>>>
    LayoutOptions(T&& device) : LayoutOptions() {
        this->set_device(std::forward<T>(device));
    }

    template <typename... Args, typename = std::enable_if_t<std::is_constructible_v<Device, Args&&...>>>
    LayoutOptions(Args&&... args) : LayoutOptions(Device(std::forward<Args>(args)...)) {}

    [[nodiscard]] LayoutOptions device(std::optional<Device> device) const noexcept {
        LayoutOptions r = *this;
        r.set_device(device);
        return r;
    }

    template <typename... Args>
    [[nodiscard]] LayoutOptions device(Args&&... args) const noexcept {
        return device(std::optional<Device>(std::in_place, std::forward<Args>(args)...));
    }

    [[nodiscard]] LayoutOptions cuda_index(int device_index) const noexcept {
        return device(Device::Type::CUDA, device_index);
    }

    [[nodiscard]] LayoutOptions dtype(std::optional<ScalarType> dtype) const noexcept {
        LayoutOptions r = *this;
        r.set_dtype(dtype);
        return r;
    }

    [[nodiscard]] LayoutOptions memory_format(std::optional<MemoryFormat> memory_format) const noexcept {
        LayoutOptions r = *this;
        r.set_memory_format(memory_format);
        return r;
    }

    void set_device(std::optional<Device> device) & noexcept {
        if (device.has_value()) {
            device_ = *device;
            has_device_ = true;
        } else {
            has_device_ = false;
        }
    }

    void set_dtype(std::optional<ScalarType> dtype) & noexcept {
        if (dtype.has_value()) {
            dtype_ = *dtype;
            has_dtype_ = true;
        } else {
            has_dtype_ = false;
        }
    }

    void set_memory_format(std::optional<MemoryFormat> memory_format) & noexcept {
        if (memory_format.has_value()) {
            memory_format_ = *memory_format;
            has_memory_format_ = true;
        } else {
            has_memory_format_ = false;
        }
    }

    bool has_device() const noexcept {
        return has_device_;
    }

    bool has_dtype() const noexcept {
        return has_dtype_;
    }

    bool has_memory_format() const noexcept {
        return has_memory_format_;
    }

    const ScalarType& dtype() const noexcept {
        return dtype_;
    }

    const Device& device() const noexcept {
        return device_;
    }

    const MemoryFormat& memory_format() const noexcept {
        return memory_format_;
    }

private:
    Device device_ = DeviceType::CPU;
    ScalarType dtype_ = ScalarType::Float;
    MemoryFormat memory_format_ = MemoryFormat::Contiguous;

    bool has_device_ = false;
    bool has_dtype_ = false;
    bool has_memory_format_ = false;
};

inline std::ostream& operator<<(std::ostream& stream, const LayoutOptions& options) {
    auto print = [&](const char* label, auto prop, bool has_prop, const char* tail = "") {
        stream << label << prop << (has_prop ? "" : " (default)") << tail;
    };

    print("LayoutOptions(dtype=", options.dtype(), options.has_dtype(), ", ");
    print("device=", options.device(), options.has_device(), ", ");
    print("memory format=", options.memory_format(), options.has_memory_format(), ")");

    return stream;
}

} // namespace witin