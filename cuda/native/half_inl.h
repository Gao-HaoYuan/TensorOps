#pragma once

#include <limits>

#ifdef __CUDACC__
#include <cuda_fp16.h>
#endif

#include "macro.h"

namespace witin {

INLINE_HOST_DEVICE Half::Half(float value)
    :
#if defined(__CUDA_ARCH__)
    x(__half_as_short(__float2half(value)))
#else
    x(host::fp16_ieee_from_fp32_value(value))
#endif
{
} 

INLINE_HOST_DEVICE Half::operator float() const {
#if defined(__CUDA_ARCH__)
    return __half2float(*reinterpret_cast<const __half*>(&x));
#else
    return host::fp16_ieee_to_fp32_value(x);
#endif
}

#if defined(__CUDACC__) || defined(__HIPCC__)
INLINE_HOST_DEVICE Half::Half(const __half& value) {
    x = *reinterpret_cast<const unsigned short*>(&value);
}
INLINE_HOST_DEVICE Half::operator __half() const {
    return *reinterpret_cast<const __half*>(&x);
}
#endif

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 350)) || (defined(__clang__) && defined(__CUDA__))
inline __device__ Half __ldg(const Half* ptr) {
    return __ldg(reinterpret_cast<const __half*>(ptr));
}
#endif

INLINE_HOST_DEVICE Half operator+(const Half& a, const Half& b) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)
    return __hadd(static_cast<__half>(a), static_cast<__half>(b));
#else
    return static_cast<float>(a) + static_cast<float>(b);
#endif
}

INLINE_HOST_DEVICE Half operator-(const Half& a, const Half& b) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)
    return __hsub(static_cast<__half>(a), static_cast<__half>(b));
#else
    return static_cast<float>(a) - static_cast<float>(b);
#endif
}

INLINE_HOST_DEVICE Half operator*(const Half& a, const Half& b) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)
    return __hmul(static_cast<__half>(a), static_cast<__half>(b));
#else
    return static_cast<float>(a) * static_cast<float>(b);
#endif
}

INLINE_HOST_DEVICE Half operator/(const Half& a, const Half& b) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)
    return __hdiv(static_cast<__half>(a), static_cast<__half>(b));
#else
    return static_cast<float>(a) / static_cast<float>(b);
#endif
}

INLINE_HOST_DEVICE Half operator-(const Half& a) {
#if (defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530)
    return __hneg(static_cast<__half>(a));
#else
    return -static_cast<float>(a);
#endif
}

INLINE_HOST_DEVICE Half& operator+=(Half& a, const Half& b) {
    a = a + b;
    return a;
}

INLINE_HOST_DEVICE Half& operator-=(Half& a, const Half& b) {
    a = a - b;
    return a;
}

INLINE_HOST_DEVICE Half& operator*=(Half& a, const Half& b) {
    a = a * b;
    return a;
}

INLINE_HOST_DEVICE Half& operator/=(Half& a, const Half& b) {
    a = a / b;
    return a;
}

// Arithmetic with floats
INLINE_HOST_DEVICE float operator+(Half a, float b) {
    return static_cast<float>(a) + b;
}
INLINE_HOST_DEVICE float operator-(Half a, float b) {
    return static_cast<float>(a) - b;
}
INLINE_HOST_DEVICE float operator*(Half a, float b) {
    return static_cast<float>(a) * b;
}
INLINE_HOST_DEVICE float operator/(Half a, float b) {
    return static_cast<float>(a) / b;
}

INLINE_HOST_DEVICE float operator+(float a, Half b) {
    return a + static_cast<float>(b);
}
INLINE_HOST_DEVICE float operator-(float a, Half b) {
    return a - static_cast<float>(b);
}
INLINE_HOST_DEVICE float operator*(float a, Half b) {
    return a * static_cast<float>(b);
}
INLINE_HOST_DEVICE float operator/(float a, Half b) {
    return a / static_cast<float>(b);
}

INLINE_HOST_DEVICE float& operator+=(float& a, const Half& b) {
    return a += static_cast<float>(b);
}
INLINE_HOST_DEVICE float& operator-=(float& a, const Half& b) {
    return a -= static_cast<float>(b);
}
INLINE_HOST_DEVICE float& operator*=(float& a, const Half& b) {
    return a *= static_cast<float>(b);
}
INLINE_HOST_DEVICE float& operator/=(float& a, const Half& b) {
    return a /= static_cast<float>(b);
}

/// Arithmetic with doubles
INLINE_HOST_DEVICE double operator+(Half a, double b) {
    return static_cast<double>(a) + b;
}
INLINE_HOST_DEVICE double operator-(Half a, double b) {
    return static_cast<double>(a) - b;
}
INLINE_HOST_DEVICE double operator*(Half a, double b) {
    return static_cast<double>(a) * b;
}
INLINE_HOST_DEVICE double operator/(Half a, double b) {
    return static_cast<double>(a) / b;
}

INLINE_HOST_DEVICE double operator+(double a, Half b) {
    return a + static_cast<double>(b);
}
INLINE_HOST_DEVICE double operator-(double a, Half b) {
    return a - static_cast<double>(b);
}
INLINE_HOST_DEVICE double operator*(double a, Half b) {
    return a * static_cast<double>(b);
}
INLINE_HOST_DEVICE double operator/(double a, Half b) {
    return a / static_cast<double>(b);
}

/// Arithmetic with ints

INLINE_HOST_DEVICE Half operator+(Half a, int b) {
  return a + static_cast<Half>(b);
}
INLINE_HOST_DEVICE Half operator-(Half a, int b) {
  return a - static_cast<Half>(b);
}
INLINE_HOST_DEVICE Half operator*(Half a, int b) {
  return a * static_cast<Half>(b);
}
INLINE_HOST_DEVICE Half operator/(Half a, int b) {
  return a / static_cast<Half>(b);
}

INLINE_HOST_DEVICE Half operator+(int a, Half b) {
    return static_cast<Half>(a) + b;
}
INLINE_HOST_DEVICE Half operator-(int a, Half b) {
    return static_cast<Half>(a) - b;
}
INLINE_HOST_DEVICE Half operator*(int a, Half b) {
    return static_cast<Half>(a) * b;
}
INLINE_HOST_DEVICE Half operator/(int a, Half b) {
    return static_cast<Half>(a) / b;
}

//// Arithmetic with int64_t
INLINE_HOST_DEVICE Half operator+(Half a, int64_t b) {
    return a + static_cast<Half>(b);
}
INLINE_HOST_DEVICE Half operator-(Half a, int64_t b) {
    return a - static_cast<Half>(b);
}
INLINE_HOST_DEVICE Half operator*(Half a, int64_t b) {
    return a * static_cast<Half>(b);
}
INLINE_HOST_DEVICE Half operator/(Half a, int64_t b) {
    return a / static_cast<Half>(b);
}

INLINE_HOST_DEVICE Half operator+(int64_t a, Half b) {
    return static_cast<Half>(a) + b;
}
INLINE_HOST_DEVICE Half operator-(int64_t a, Half b) {
    return static_cast<Half>(a) - b;
}
INLINE_HOST_DEVICE Half operator*(int64_t a, Half b) {
    return static_cast<Half>(a) * b;
}
INLINE_HOST_DEVICE Half operator/(int64_t a, Half b) {
    return static_cast<Half>(a) / b;
}

} // end of namespace witin

namespace std {

template <>
class numeric_limits<witin::Half> {
public:
    static constexpr bool is_specialized = true;
    static constexpr bool is_signed = true;
    static constexpr bool is_integer = false;
    static constexpr bool is_exact = false;
    static constexpr bool has_infinity = true;
    static constexpr bool has_quiet_NaN = true;
    static constexpr bool has_signaling_NaN = true;
    static constexpr auto has_denorm = numeric_limits<float>::has_denorm;
    static constexpr auto has_denorm_loss = numeric_limits<float>::has_denorm_loss;
    static constexpr auto round_style = numeric_limits<float>::round_style;
    static constexpr bool is_iec559 = true;
    static constexpr bool is_bounded = true;
    static constexpr bool is_modulo = false;
    static constexpr int digits = 11;
    static constexpr int digits10 = 3;
    static constexpr int max_digits10 = 5;
    static constexpr int radix = 2;
    static constexpr int min_exponent = -13;
    static constexpr int min_exponent10 = -4;
    static constexpr int max_exponent = 16;
    static constexpr int max_exponent10 = 4;
    static constexpr auto traps = numeric_limits<float>::traps;
    static constexpr auto tinyness_before = numeric_limits<float>::tinyness_before;

    static constexpr witin::Half min() { return witin::Half(0x0400, witin::Half::from_bits()); }
    static constexpr witin::Half lowest() { return witin::Half(0xFBFF, witin::Half::from_bits()); }
    static constexpr witin::Half max() { return witin::Half(0x7BFF, witin::Half::from_bits()); }
    static constexpr witin::Half epsilon() { return witin::Half(0x1400, witin::Half::from_bits()); }
    static constexpr witin::Half round_error() { return witin::Half(0x3800, witin::Half::from_bits()); }
    static constexpr witin::Half infinity() { return witin::Half(0x7C00, witin::Half::from_bits()); }
    static constexpr witin::Half quiet_NaN() { return witin::Half(0x7E00, witin::Half::from_bits()); }
    static constexpr witin::Half signaling_NaN() { return witin::Half(0x7D00, witin::Half::from_bits()); }
    static constexpr witin::Half denorm_min() { return witin::Half(0x0001, witin::Half::from_bits()); }
};

} // namespace std