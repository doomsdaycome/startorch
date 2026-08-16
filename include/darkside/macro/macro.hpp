#ifndef DARKSIDE_CORE_MACRO_HPP_
#define DARKSIDE_CORE_MACRO_HPP_

namespace darkside {

#define DARKSIDE_FORALL_SCALAR_TYPE(MACRO) \
  MACRO(ScalarType::kBool)                 \
                                           \
  MACRO(ScalarType::kUnsignedInt8)         \
  MACRO(ScalarType::kUnsignedInt16)        \
  MACRO(ScalarType::kUnsignedInt32)        \
  MACRO(ScalarType::kUnsignedInt64)        \
                                           \
  MACRO(ScalarType::kInt8)                 \
  MACRO(ScalarType::kInt16)                \
  MACRO(ScalarType::kInt32)                \
  MACRO(ScalarType::kInt64)                \
                                           \
  MACRO(ScalarType::kFloat16)              \
  MACRO(ScalarType::kFloat32)              \
  MACRO(ScalarType::kFloat64)              \
  MACRO(ScalarType::kFloat128)             \
                                           \
  MACRO(ScalarType::kBrainFloat16_t)

#define DARKSIDE_FORALL_CPP_TYPE(MACRO) \
  MACRO(bool)                           \
                                        \
  MACRO(uint8_t)                        \
  MACRO(uint16_t)                       \
  MACRO(uint32_t)                       \
  MACRO(uint64_t)                       \
                                        \
  MACRO(int8_t)                         \
  MACRO(int16_t)                        \
  MACRO(int32_t)                        \
  MACRO(int64_t)                        \
                                        \
  MACRO(std::float16_t)                 \
  MACRO(std::float32_t)                 \
  MACRO(std::float64_t)                 \
  MACRO(std::float128_t)                \
                                        \
  MACRO(std::bfloat16_t)

#define DARKSIDE_FORALL_SCALAR_CPP_TYPE(MACRO)  \
  MACRO(ScalarType::kBool, bool)                \
                                                \
  MACRO(ScalarType::kUnsignedInt8, uint8_t)     \
  MACRO(ScalarType::kUnsignedInt16, uint16_t)   \
  MACRO(ScalarType::kUnsignedInt32, uint32_t)   \
  MACRO(ScalarType::kUnsignedInt64, uint64_t)   \
                                                \
  MACRO(ScalarType::kInt8, int8_t)              \
  MACRO(ScalarType::kInt16, int16_t)            \
  MACRO(ScalarType::kInt32, int32_t)            \
  MACRO(ScalarType::kInt64, int64_t)            \
                                                \
  MACRO(ScalarType::kFloat16, std::float16_t)   \
  MACRO(ScalarType::kFloat32, std::float32_t)   \
  MACRO(ScalarType::kFloat64, std::float64_t)   \
  MACRO(ScalarType::kFloat128, std::float128_t) \
                                                \
  MACRO(ScalarType::kBrainFloat16_t, std::bfloat16_t)

#define DARKSIDE_FORALL_CPP_SCALAR_TYPE(MACRO)  \
  MACRO(bool, ScalarType::kBool)                \
                                                \
  MACRO(uint8_t, ScalarType::kUnsignedInt8)     \
  MACRO(uint16_t, ScalarType::kUnsignedInt16)   \
  MACRO(uint32_t, ScalarType::kUnsignedInt32)   \
  MACRO(uint64_t, ScalarType::kUnsignedInt64)   \
                                                \
  MACRO(int8_t, ScalarType::kInt8)              \
  MACRO(int16_t, ScalarType::kInt16)            \
  MACRO(int32_t, ScalarType::kInt32)            \
  MACRO(int64_t, ScalarType::kInt64)            \
                                                \
  MACRO(std::float16_t, ScalarType::kFloat16)   \
  MACRO(std::float32_t, ScalarType::kFloat32)   \
  MACRO(std::float64_t, ScalarType::kFloat64)   \
  MACRO(std::float128_t, ScalarType::kFloat128) \
                                                \
  MACRO(std::bfloat16_t, ScalarType::kBrainFloat16_t)

}  // namespace darkside

#endif  // DARKSIDE_CORE_MACRO_HPP_
