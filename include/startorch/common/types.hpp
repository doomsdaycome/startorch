#ifndef STARTORCH_COMMON_TYPES_HPP_
#define STARTORCH_COMMON_TYPES_HPP_

#include <cstdint>
#include <variant>

#if __has_include(<stdfloat>)
#include <stdfloat>
#endif

namespace startorch {

enum class SystemType : std::uint8_t {
  kUndefined = 0,

  kHost = 1,
  kDevice = 2,

  kOptionCount = 3
};

enum class BufferType : std::uint8_t {
  kUndefined = 0,

  kHost = 1,
  kDevice = 2,
  kPinned = 3,
  kUnified = 4,

};

enum class BridgeType : std::uint8_t {
  kUndefined = 0,

  kHostToHost = 1,
  kHostToDevice = 2,
  kDeviceToHost = 3,
  kDeviceToDevice = 4,

  kOptionmCount = 5,
};

enum class ScalarType : std::uint8_t {
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

using CppType = std::variant<std::monostate, bool, std::uint8_t, std::uint16_t,
                             std::uint32_t, std::uint64_t, std::int8_t,
                             std::int16_t, std::int32_t, std::int64_t

#if defined(__STDCPP_FLOAT16_T__)
                             ,
                             std::float16_t, std::float32_t, std::float64_t,
                             std::float128_t, std::bfloat16_t
#else
                             ,
                             float, double
#endif
                             >;

} // namespace startorch

#endif // !STARTORCH_COMMON_TYPES_HPP_
