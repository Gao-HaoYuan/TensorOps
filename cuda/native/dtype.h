#pragma once

#include<string>

#include "macro.h"
#include "half.h"

namespace witin {

#define FORALL_SCALAR_TYPES(_)                      \
    _(uint8_t, Byte) /* 0 */                        \
    _(int8_t, Char) /* 1 */                         \
    _(int16_t, Short) /* 2 */                       \
    _(int, Int) /* 3 */                             \
    _(int64_t, Long) /* 4 */                        \
    _(witin::Half, Half) /* 5 */                    \
    _(float, Float) /* 6 */                         \
    _(double, Double) /* 7 */                       \
    _(bool, Bool) /* 8 */                           \
    _(uint16_t, UInt16) /* 9 */                     \
    _(uint32_t, UInt32) /* 10 */                    \
    _(uint64_t, UInt64) /* 11 */

enum class ScalarType : int8_t {
#define DEFINE_ST_ENUM_VAL_(_1, n) n,
    FORALL_SCALAR_TYPES(DEFINE_ST_ENUM_VAL_)
#undef DEFINE_ST_ENUM_VAL_
    Undefined,
    NumOptions
};

constexpr uint16_t NumScalarTypes = static_cast<uint16_t>(ScalarType::NumOptions);

template <ScalarType N>
struct ScalarTypeToCPPType;

#define SPECIALIZE_ScalarTypeToCPPType(cpp_type, scalar_type)           \
    template <>                                                         \
    struct ScalarTypeToCPPType<ScalarType::scalar_type> {               \
        using type = cpp_type;                                          \
        static type t;                                                  \
    };

FORALL_SCALAR_TYPES(SPECIALIZE_ScalarTypeToCPPType)
#undef SPECIALIZE_ScalarTypeToCPPType

template <ScalarType N>
using ScalarTypeToCPPTypeT = typename ScalarTypeToCPPType<N>::type;

template <typename T>
struct CppTypeToScalarType;

#define SPECIALIZE_CppTypeToScalarType(cpp_type, scalar_type)                           \
    template <>                                                                         \
    struct CppTypeToScalarType<cpp_type>                                                \
        : std::                                                                         \
            integral_constant<witin::ScalarType, witin::ScalarType::scalar_type> {      \
    };

FORALL_SCALAR_TYPES(SPECIALIZE_CppTypeToScalarType)
#undef SPECIALIZE_CppTypeToScalarType

inline const char* toString(ScalarType t) {
#define DEFINE_CASE(_, name)    \
    case ScalarType::name:     \
        return #name;

    switch (t) {
        FORALL_SCALAR_TYPES(DEFINE_CASE)
        default:
            return "UNKNOWN_SCALAR";
    }
#undef DEFINE_CASE
}

inline size_t elementSize(ScalarType t) {
#define CASE_ELEMENTSIZE_CASE(ctype, name)      \
    case ScalarType::name:                      \
        return sizeof(ctype);

    switch (t) {
        FORALL_SCALAR_TYPES(CASE_ELEMENTSIZE_CASE)
        default:
            ASSERT(false, "Unknown ScalarType");
    }
#undef CASE_ELEMENTSIZE_CASE
}

inline bool isIntegralType(ScalarType t, bool includeBool) {
    bool isIntegral =
        (t == ScalarType::Byte || t == ScalarType::Char || t == ScalarType::Int ||
         t == ScalarType::Long || t == ScalarType::Short ||
         t == ScalarType::UInt16 || t == ScalarType::UInt32 ||
         t == ScalarType::UInt64);

    return isIntegral || (includeBool && t == ScalarType::Bool);
}

inline bool isReducedFloatingType(ScalarType t) {
    return t == ScalarType::Half;
}

inline bool isFloatingType(ScalarType t) {
    return t == ScalarType::Double || t == ScalarType::Float || isReducedFloatingType(t);
}

inline bool isSignedType(ScalarType t) {
#define CASE_ISSIGNED(_, name)              \
    case ScalarType::name:                  \
        return std::numeric_limits<         \
            ScalarTypeToCPPTypeT<ScalarType::name>>::is_signed;

    switch (t) {
        FORALL_SCALAR_TYPES(CASE_ISSIGNED)
        default:
            return "UNKNOWN_SCALAR";
    }
#undef CASE_ISSIGNED
}

inline std::ostream& operator<<(std::ostream& stream, witin::ScalarType scalar_type) {
    return stream << toString(scalar_type);
}

} // // end of namespace witin