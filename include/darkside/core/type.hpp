#ifndef DARKSIDE_CORE_TYPES_HPP_
#define DARKSIDE_CORE_TYPES_HPP_

#include <cstdint>
#include <stdfloat>
#include <variant>

namespace darkside {

enum class ProcessingUnitType : uint8_t {
  kUndefined = 0,

  kCentral = 1,
  kGraphics = 2,

  kOptionCount = 3
};

enum class RandomAccessMemoryType : uint8_t {
  kUndefined = 0,

  kHost = 1,
  kPinned = 2,
  kDevice = 3,
  kUnified = 4,

  kOptionCount = 5
};

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

  kOptionCount = 13
};

using ValueType = std::variant<bool, uint8_t, uint16_t, uint32_t, uint64_t,
                               int8_t, int16_t, int32_t, int64_t,
                               std::float16_t, std::float32_t, std::float64_t>;

}  // namespace darkside

#endif  // !DARKSIDE_CORE_TYPES_HPP_
