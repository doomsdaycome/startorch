#ifndef STARTORCH_COMMON_TYPES_HPP_
#define STARTORCH_COMMON_TYPES_HPP_

#include <cstdint>
#include <variant>

#if __has_include(<stdfloat>)
#include <stdfloat>
#define STARTORCH_COMMON_STDFLOAT_HPP_
#endif

namespace darkside {

#if defined(__cpp_lib_stdfloat)

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

  kFloat16 = 10,
  kFloat32 = 11,
  kFloat64 = 12,
  kFloat128 = 13,

  kBrainFloat16 = 14,

  kOptionCount = 15
};

using CPPType = std::variant<std::monostate, bool, uint8_t, uint16_t, uint32_t,
                             uint64_t, int8_t, int16_t, int32_t, int64_t,
                             std::float16_t, std::float32_t, std::float64_t,
                             std::float128_t, std::bfloat16_t>;

#else

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

  kFloat32 = 10,
  kFloat64 = 11,

  kOptionCount = 12
};

using CPPType =
    std::variant<std::monostate, bool, uint8_t, uint16_t, uint32_t, uint64_t,
                 int8_t, int16_t, int32_t, int64_t, float, double>;

#endif

}  // namespace darkside

#endif  // !STARTORCH_COMMON_TYPES_HPP_
