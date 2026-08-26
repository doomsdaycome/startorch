#ifndef DARKSIDE_MACROS_MAPPING_HPP_
#define DARKSIDE_MACROS_MAPPING_HPP_

#if defined(__STDCPP_FLOAT16_T__)
#define DARKSIDE_FORALL_SCALAR_TYPE(MACRO)                                     \
  MACRO(::startorch::ScalarType::kBool)                                        \
                                                                               \
  MACRO(::startorch::ScalarType::kUnsignedInt8)                                \
  MACRO(::startorch::ScalarType::kUnsignedInt16)                               \
  MACRO(::startorch::ScalarType::kUnsignedInt32)                               \
  MACRO(::startorch::ScalarType::kUnsignedInt64)                               \
                                                                               \
  MACRO(::startorch::ScalarType::kInt8)                                        \
  MACRO(::startorch::ScalarType::kInt16)                                       \
  MACRO(::startorch::ScalarType::kInt32)                                       \
  MACRO(::startorch::ScalarType::kInt64)                                       \
                                                                               \
  MACRO(::startorch::ScalarType::kFloat16)                                     \
  MACRO(::startorch::ScalarType::kFloat32)                                     \
  MACRO(::startorch::ScalarType::kFloat64)                                     \
  MACRO(::startorch::ScalarType::kFloat128)                                    \
                                                                               \
  MACRO(::startorch::ScalarType::kBrainFloat16)
#else
#define DARKSIDE_FORALL_SCALAR_TYPE(MACRO)                                     \
  MACRO(::startorch::ScalarType::kBool)                                        \
                                                                               \
  MACRO(::startorch::ScalarType::kUnsignedInt8)                                \
  MACRO(::startorch::ScalarType::kUnsignedInt16)                               \
  MACRO(::startorch::ScalarType::kUnsignedInt32)                               \
  MACRO(::startorch::ScalarType::kUnsignedInt64)                               \
                                                                               \
  MACRO(::startorch::ScalarType::kInt8)                                        \
  MACRO(::startorch::ScalarType::kInt16)                                       \
  MACRO(::startorch::ScalarType::kInt32)                                       \
  MACRO(::startorch::ScalarType::kInt64)                                       \
                                                                               \
  MACRO(::startorch::ScalarType::kFloat32)                                     \
  MACRO(::startorch::ScalarType::kFloat64)
#endif

#if defined(__STDCPP_FLOAT16_T__)
#define DARKSIDE_FORALL_CPP_TYPE(MACRO)                                        \
  MACRO(bool)                                                                  \
                                                                               \
  MACRO(uint8_t)                                                               \
  MACRO(uint16_t)                                                              \
  MACRO(uint32_t)                                                              \
  MACRO(uint64_t)                                                              \
                                                                               \
  MACRO(int8_t)                                                                \
  MACRO(int16_t)                                                               \
  MACRO(int32_t)                                                               \
  MACRO(int64_t)                                                               \
                                                                               \
  MACRO(float16_t)                                                             \
  MACRO(float32_t)                                                             \
  MACRO(float64_t)                                                             \
  MACRO(float128_t)                                                            \
                                                                               \
  MACRO(bfloat16_t)
#else
#define DARKSIDE_FORALL_CPP_TYPE(MACRO)                                        \
  MACRO(bool)                                                                  \
                                                                               \
  MACRO(uint8_t)                                                               \
  MACRO(uint16_t)                                                              \
  MACRO(uint32_t)                                                              \
  MACRO(uint64_t)                                                              \
                                                                               \
  MACRO(int8_t)                                                                \
  MACRO(int16_t)                                                               \
  MACRO(int32_t)                                                               \
  MACRO(int64_t)                                                               \
                                                                               \
  MACRO(float)                                                                 \
  MACRO(double)
#endif

#if defined(__STDCPP_FLOAT16_T__)
#define DARKSIDE_FORALL_SCALAR_TYPE_TO_CPP_TYPE(MACRO)                         \
  MACRO(::startorch::ScalarType::kUndefined, std::monostate)                   \
  MACRO(::startorch::ScalarType::kBool, bool)                                  \
                                                                               \
  MACRO(::startorch::ScalarType::kUnsignedInt8, uint8_t)                       \
  MACRO(::startorch::ScalarType::kUnsignedInt16, uint16_t)                     \
  MACRO(::startorch::ScalarType::kUnsignedInt32, uint32_t)                     \
  MACRO(::startorch::ScalarType::kUnsignedInt64, uint64_t)                     \
                                                                               \
  MACRO(::startorch::ScalarType::kInt8, int8_t)                                \
  MACRO(::startorch::ScalarType::kInt16, int16_t)                              \
  MACRO(::startorch::ScalarType::kInt32, int32_t)                              \
  MACRO(::startorch::ScalarType::kInt64, int64_t)                              \
                                                                               \
  MACRO(::startorch::ScalarType::kFloat16, float16_t)                          \
  MACRO(::startorch::ScalarType::kFloat32, float32_t)                          \
  MACRO(::startorch::ScalarType::kFloat64, float64_t)                          \
  MACRO(::startorch::ScalarType::kFloat128, float128_t)                        \
                                                                               \
  MACRO(::startorch::ScalarType::kBrainFloat16, bfloat16_t)
#else
#define DARKSIDE_FORALL_SCALAR_TYPE_TO_CPP_TYPE(MACRO)                         \
  MACRO(::startorch::ScalarType::kUndefined, std::monostate)                   \
  MACRO(::startorch::ScalarType::kBool, bool)                                  \
                                                                               \
  MACRO(::startorch::ScalarType::kUnsignedInt8, uint8_t)                       \
  MACRO(::startorch::ScalarType::kUnsignedInt16, uint16_t)                     \
  MACRO(::startorch::ScalarType::kUnsignedInt32, uint32_t)                     \
  MACRO(::startorch::ScalarType::kUnsignedInt64, uint64_t)                     \
                                                                               \
  MACRO(::startorch::ScalarType::kInt8, int8_t)                                \
  MACRO(::startorch::ScalarType::kInt16, int16_t)                              \
  MACRO(::startorch::ScalarType::kInt32, int32_t)                              \
  MACRO(::startorch::ScalarType::kInt64, int64_t)                              \
                                                                               \
  MACRO(::startorch::ScalarType::kFloat32, float)                              \
  MACRO(::startorch::ScalarType::kFloat64, double)
#endif

#if defined(__STDCPP_FLOAT16_T__)
#define DARKSIDE_FORALL_CPP_TYPE_TO_SCALAR_TYPE(MACRO)                         \
  MACRO(std::monostate, ::startorch::ScalarType::kUndefined)                   \
  MACRO(bool, ::startorch::ScalarType::kBool)                                  \
                                                                               \
  MACRO(uint8_t, ::startorch::ScalarType::kUnsignedInt8)                       \
  MACRO(uint16_t, ::startorch::ScalarType::kUnsignedInt16)                     \
  MACRO(uint32_t, ::startorch::ScalarType::kUnsignedInt32)                     \
  MACRO(uint64_t, ::startorch::ScalarType::kUnsignedInt64)                     \
                                                                               \
  MACRO(int8_t, ::startorch::ScalarType::kInt8)                                \
  MACRO(int16_t, ::startorch::ScalarType::kInt16)                              \
  MACRO(int32_t, ::startorch::ScalarType::kInt32)                              \
  MACRO(int64_t, ::startorch::ScalarType::kInt64)                              \
                                                                               \
  MACRO(float16_t, ::startorch::ScalarType::kFloat16)                          \
  MACRO(float32_t, ::startorch::ScalarType::kFloat32)                          \
  MACRO(float64_t, ::startorch::ScalarType::kFloat64)                          \
  MACRO(float128_t, ::startorch::ScalarType::kFloat128)                        \
                                                                               \
  MACRO(bfloat16_t, ::startorch::ScalarType::kBrainFloat16)
#else
#define DARKSIDE_FORALL_CPP_TYPE_TO_SCALAR_TYPE(MACRO)                         \
  MACRO(std::monostate, ::startorch::ScalarType::kUndefined)                   \
  MACRO(bool, ::startorch::ScalarType::kBool)                                  \
                                                                               \
  MACRO(uint8_t, ::startorch::ScalarType::kUnsignedInt8)                       \
  MACRO(uint16_t, ::startorch::ScalarType::kUnsignedInt16)                     \
  MACRO(uint32_t, ::startorch::ScalarType::kUnsignedInt32)                     \
  MACRO(uint64_t, ::startorch::ScalarType::kUnsignedInt64)                     \
                                                                               \
  MACRO(int8_t, ::startorch::ScalarType::kInt8)                                \
  MACRO(int16_t, ::startorch::ScalarType::kInt16)                              \
  MACRO(int32_t, ::startorch::ScalarType::kInt32)                              \
  MACRO(int64_t, ::startorch::ScalarType::kInt64)                              \
                                                                               \
  MACRO(float, ::startorch::ScalarType::kFloat32)                              \
  MACRO(double, ::startorch::ScalarType::kFloat64)
#endif

#endif // !DARKSIDE_MACROS_MAPPING_HPP_
