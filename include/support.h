#pragma once

#include <type_traits>
#include <string>

#include <torch/types.h>

#include "half.h"

// get device of first tensor param
template <typename T, typename... Args>
std::enable_if_t<std::is_same<std::decay_t<T>, at::Tensor>::value, at::Device>
GetFirstTensorDevice(T&& t, Args&&... args) {
    return std::forward<T>(t).device();
}

template <typename T, typename... Args>
std::enable_if_t<!std::is_same<std::decay_t<T>, at::Tensor>::value, at::Device>
GetFirstTensorDevice(T&& t, Args&&... args) {
    return GetFirstTensorDevice(std::forward<Args>(args)...);
}

// check device consistency
inline std::pair<int, at::Device> 
CheckDeviceConsistency(const at::Device& device, int index) {
    return {index, device};
}

template <typename T, typename... Args>
std::enable_if_t<std::is_same<std::decay_t<T>, at::Tensor>::value, std::pair<int, at::Device>>
CheckDeviceConsistency(const at::Device& device, int index, T&& t, Args&&... args) {
    auto new_device = std::forward<T>(t).device();
    if (new_device.type() != device.type() || new_device.index() != device.index()) {
        return {index, new_device};
    }
    return CheckDeviceConsistency(device, index + 1, std::forward<Args>(args)...);
}

template <typename T, typename... Args>
std::enable_if_t<!std::is_same<std::decay_t<T>, at::Tensor>::value, std::pair<int, at::Device>>
CheckDeviceConsistency(const at::Device& device, int index, T&& t, Args&&... args) {
    return CheckDeviceConsistency(device, index + 1, std::forward<Args>(args)...);
}

template <typename... Args>
void CheckConsistency(const char* name, Args&&... args) {
    auto device = GetFirstTensorDevice(args...);
    auto inconsist = CheckDeviceConsistency(device, 0, args...);
    TORCH_CHECK(inconsist.first >= sizeof...(args), name,  ": at param ", inconsist.first, 
                ", inconsistent device: ", inconsist.second.str(), " vs ", device.str(), "\n");
}

template<typename U> struct native_type; 

template<> struct native_type<uint8_t> { using T = uint8_t; }; 
template<> struct native_type<int8_t> { using T = int8_t; }; 
template<> struct native_type<int32_t> { using T = int32_t; }; 
template<> struct native_type<int64_t> { using T = int64_t; };
template<> struct native_type<float> { using T = float; }; 
template<> struct native_type<double> { using T = double; };  
template<> struct native_type<c10::Half> { using T = witin::Half; }; 

template<typename U> 
using native_t = typename native_type<U>::T;