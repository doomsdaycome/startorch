#ifndef STARTORCH_COMMON_TYPES_HPP_
#define STARTORCH_COMMON_TYPES_HPP_

#include <cstdint>
#include <variant>

#if __has_include(<stdfloat>)
#include <stdfloat>
#endif

namespace startorch {

enum class ScalarType : uint8_t {
  kUndefined = 0,

  kBool = 1,

  kUnsignedInt8 = 2,
  kUnsignedInt16 = 3,
  kUnsignedInt32 = 4,
  kUnsignedInt64 = 5,

  kInt8 = 6,
  kInt16 = 7,
  kInt32 = 8,
  kInt64 = 9,

#if defined(__STDCPP_FLOAT16_T__)
  kFloat16 = 10,
  kFloat32 = 11,
  kFloat64 = 12,
  kFloat128 = 13,

  kBrainFloat16 = 14,

  kOptionCount = 15
#else
  kFloat32 = 10,
  kFloat64 = 11,

  kOptionCount = 12
#endif
};

using CppType =
    std::variant<std::monostate, bool, uint8_t, uint16_t, uint32_t, uint64_t,
                 int8_t, int16_t, int32_t, int64_t

#if defined(__STDCPP_FLOAT16_T__)
                 ,
                 float16_t, float32_t, float64_t, float128_t, bfloat16_t
#else
                 ,
                 float, double
#endif
                 >;

} // namespace startorch

#endif // STARTORCH_COMMON_TYPES_HPP_
