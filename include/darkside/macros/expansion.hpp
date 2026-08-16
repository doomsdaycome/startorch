#ifndef DARKSIDE_MACROS_EXPANSION_HPP_
#define DARKSIDE_MACROS_EXPANSION_HPP_

namespace darkside {

#define DARKSIDE_FORALL_SCALAR_TYPE(MACRO)      \
  MACRO(::darkside::ScalarType::kBool)          \
                                                \
  MACRO(::darkside::ScalarType::kUnsignedInt8)  \
  MACRO(::darkside::ScalarType::kUnsignedInt16) \
  MACRO(::darkside::ScalarType::kUnsignedInt32) \
  MACRO(::darkside::ScalarType::kUnsignedInt64) \
                                                \
  MACRO(::darkside::ScalarType::kInt8)          \
  MACRO(::darkside::ScalarType::kInt16)         \
  MACRO(::darkside::ScalarType::kInt32)         \
  MACRO(::darkside::ScalarType::kInt64)         \
                                                \
  MACRO(::darkside::ScalarType::kFloat16)       \
  MACRO(::darkside::ScalarType::kFloat32)       \
  MACRO(::darkside::ScalarType::kFloat64)       \
  MACRO(::darkside::ScalarType::kFloat128)      \
                                                \
  MACRO(::darkside::ScalarType::kBrainFloat16_t)

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

#define DARKSIDE_FORALL_SCALAR_CPP_TYPE(MACRO)              \
  MACRO(::darkside::ScalarType::kBool, bool)                \
                                                            \
  MACRO(::darkside::ScalarType::kUnsignedInt8, uint8_t)     \
  MACRO(::darkside::ScalarType::kUnsignedInt16, uint16_t)   \
  MACRO(::darkside::ScalarType::kUnsignedInt32, uint32_t)   \
  MACRO(::darkside::ScalarType::kUnsignedInt64, uint64_t)   \
                                                            \
  MACRO(::darkside::ScalarType::kInt8, int8_t)              \
  MACRO(::darkside::ScalarType::kInt16, int16_t)            \
  MACRO(::darkside::ScalarType::kInt32, int32_t)            \
  MACRO(::darkside::ScalarType::kInt64, int64_t)            \
                                                            \
  MACRO(::darkside::ScalarType::kFloat16, std::float16_t)   \
  MACRO(::darkside::ScalarType::kFloat32, std::float32_t)   \
  MACRO(::darkside::ScalarType::kFloat64, std::float64_t)   \
  MACRO(::darkside::ScalarType::kFloat128, std::float128_t) \
                                                            \
  MACRO(::darkside::ScalarType::kBrainFloat16_t, std::bfloat16_t)

#define DARKSIDE_FORALL_CPP_SCALAR_TYPE(MACRO)              \
  MACRO(bool, ::darkside::ScalarType::kBool)                \
                                                            \
  MACRO(uint8_t, ::darkside::ScalarType::kUnsignedInt8)     \
  MACRO(uint16_t, ::darkside::ScalarType::kUnsignedInt16)   \
  MACRO(uint32_t, ::darkside::ScalarType::kUnsignedInt32)   \
  MACRO(uint64_t, ::darkside::ScalarType::kUnsignedInt64)   \
                                                            \
  MACRO(int8_t, ::darkside::ScalarType::kInt8)              \
  MACRO(int16_t, ::darkside::ScalarType::kInt16)            \
  MACRO(int32_t, ::darkside::ScalarType::kInt32)            \
  MACRO(int64_t, ::darkside::ScalarType::kInt64)            \
                                                            \
  MACRO(std::float16_t, ::darkside::ScalarType::kFloat16)   \
  MACRO(std::float32_t, ::darkside::ScalarType::kFloat32)   \
  MACRO(std::float64_t, ::darkside::ScalarType::kFloat64)   \
  MACRO(std::float128_t, ::darkside::ScalarType::kFloat128) \
                                                            \
  MACRO(std::bfloat16_t, ::darkside::ScalarType::kBrainFloat16_t)

}  // namespace darkside

#endif  // DARKSIDE_MACROS_EXPANSION_HPP_
