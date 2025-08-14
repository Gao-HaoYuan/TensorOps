#pragma once

#include <cstdint>


#if __has_include(<bit>) && (defined(__cpp_lib_bit_cast) && __cpp_lib_bit_cast >= 201806L)
#include <bit>
#define HAVE_STD_BIT_CAST 1
#else
#define HAVE_STD_BIT_CAST 0
#endif

#include "macro.h"


namespace witin::host {

#if HAVE_STD_BIT_CAST
using std::bit_cast;
#else
// Implementations of std::bit_cast() from C++ 20.
//
// This is a less sketchy version of reinterpret_cast.
//
// See https://en.cppreference.com/w/cpp/numeric/bit_cast for more
// information as well as the source of our implementations.
template <class To, class From>
INLINE_HOST_DEVICE std::enable_if_t<
    sizeof(To) == sizeof(From) && std::is_trivially_copyable_v<From> &&
        std::is_trivially_copyable_v<To>,
    To>
// constexpr support needs compiler magic
bit_cast(const From& src) noexcept {
  static_assert(
      std::is_trivially_constructible_v<To>,
      "This implementation additionally requires "
      "destination type to be trivially constructible");

  To dst;
  std::memcpy(&dst, &src, sizeof(To));
  return dst;
}
#endif
#undef HAVE_STD_BIT_CAST

INLINE_HOST_DEVICE float fp32_from_bits(uint32_t w) {
#if defined(__CUDA_ARCH__)
  return __uint_as_float((unsigned int)w);
#else
  return bit_cast<float>(w);
#endif
}

INLINE_HOST_DEVICE uint32_t fp32_to_bits(float f) {
#if defined(__CUDA_ARCH__)
  return (uint32_t)__float_as_uint(f);
#else
  return bit_cast<uint32_t>(f);
#endif
}

} // namespace witin::host
