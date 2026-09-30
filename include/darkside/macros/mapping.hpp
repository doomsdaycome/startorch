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
  MACRO(std::uint8_t)                                                          \
  MACRO(std::uint16_t)                                                         \
  MACRO(std::uint32_t)                                                         \
  MACRO(std::uint64_t)                                                         \
                                                                               \
  MACRO(std::int8_t)                                                           \
  MACRO(std::int16_t)                                                          \
  MACRO(std::int32_t)                                                          \
  MACRO(std::int64_t)                                                          \
                                                                               \
  MACRO(std::float16_t)                                                        \
  MACRO(std::float32_t)                                                        \
  MACRO(std::float64_t)                                                        \
  MACRO(std::float128_t)                                                       \
                                                                               \
  MACRO(std::bfloat16_t)
#else
#define DARKSIDE_FORALL_CPP_TYPE(MACRO)                                        \
  MACRO(bool)                                                                  \
                                                                               \
  MACRO(std::uint8_t)                                                          \
  MACRO(std::uint16_t)                                                         \
  MACRO(std::uint32_t)                                                         \
  MACRO(std::uint64_t)                                                         \
                                                                               \
  MACRO(std::int8_t)                                                           \
  MACRO(std::int16_t)                                                          \
  MACRO(std::int32_t)                                                          \
  MACRO(std::int64_t)                                                          \
                                                                               \
  MACRO(float)                                                                 \
  MACRO(double)
#endif

#if defined(__STDCPP_FLOAT16_T__)
#define DARKSIDE_FORALL_SCALAR_TYPE_TO_CPP_TYPE(MACRO)                         \
  MACRO(::startorch::ScalarType::kUndefined, std::monostate)                   \
  MACRO(::startorch::ScalarType::kBool, bool)                                  \
                                                                               \
  MACRO(::startorch::ScalarType::kUnsignedInt8, std::uint8_t)                  \
  MACRO(::startorch::ScalarType::kUnsignedInt16, std::uint16_t)                \
  MACRO(::startorch::ScalarType::kUnsignedInt32, std::uint32_t)                \
  MACRO(::startorch::ScalarType::kUnsignedInt64, std::uint64_t)                \
                                                                               \
  MACRO(::startorch::ScalarType::kInt8, std::int8_t)                           \
  MACRO(::startorch::ScalarType::kInt16, std::int16_t)                         \
  MACRO(::startorch::ScalarType::kInt32, std::int32_t)                         \
  MACRO(::startorch::ScalarType::kInt64, std::int64_t)                         \
                                                                               \
  MACRO(::startorch::ScalarType::kFloat16, std::float16_t)                     \
  MACRO(::startorch::ScalarType::kFloat32, std::float32_t)                     \
  MACRO(::startorch::ScalarType::kFloat64, std::float64_t)                     \
  MACRO(::startorch::ScalarType::kFloat128, std::float128_t)                   \
                                                                               \
  MACRO(::startorch::ScalarType::kBrainFloat16, std::bfloat16_t)
#else
#define DARKSIDE_FORALL_SCALAR_TYPE_TO_CPP_TYPE(MACRO)                         \
  MACRO(::startorch::ScalarType::kUndefined, std::monostate)                   \
  MACRO(::startorch::ScalarType::kBool, bool)                                  \
                                                                               \
  MACRO(::startorch::ScalarType::kUnsignedInt8, std::uint8_t)                  \
  MACRO(::startorch::ScalarType::kUnsignedInt16, std::uint16_t)                \
  MACRO(::startorch::ScalarType::kUnsignedInt32, std::uint32_t)                \
  MACRO(::startorch::ScalarType::kUnsignedInt64, std::uint64_t)                \
                                                                               \
  MACRO(::startorch::ScalarType::kInt8, std::int8_t)                           \
  MACRO(::startorch::ScalarType::kInt16, std::int16_t)                         \
  MACRO(::startorch::ScalarType::kInt32, std::int32_t)                         \
  MACRO(::startorch::ScalarType::kInt64, std::int64_t)                         \
                                                                               \
  MACRO(::startorch::ScalarType::kFloat32, float)                              \
  MACRO(::startorch::ScalarType::kFloat64, double)
#endif

#if defined(__STDCPP_FLOAT16_T__)
#define DARKSIDE_FORALL_CPP_TYPE_TO_SCALAR_TYPE(MACRO)                         \
  MACRO(std::monostate, ::startorch::ScalarType::kUndefined)                   \
  MACRO(bool, ::startorch::ScalarType::kBool)                                  \
                                                                               \
  MACRO(std::uint8_t, ::startorch::ScalarType::kUnsignedInt8)                  \
  MACRO(std::uint16_t, ::startorch::ScalarType::kUnsignedInt16)                \
  MACRO(std::uint32_t, ::startorch::ScalarType::kUnsignedInt32)                \
  MACRO(std::uint64_t, ::startorch::ScalarType::kUnsignedInt64)                \
                                                                               \
  MACRO(std::int8_t, ::startorch::ScalarType::kInt8)                           \
  MACRO(std::int16_t, ::startorch::ScalarType::kInt16)                         \
  MACRO(std::int32_t, ::startorch::ScalarType::kInt32)                         \
  MACRO(std::int64_t, ::startorch::ScalarType::kInt64)                         \
                                                                               \
  MACRO(std::float16_t, ::startorch::ScalarType::kFloat16)                     \
  MACRO(std::float32_t, ::startorch::ScalarType::kFloat32)                     \
  MACRO(std::float64_t, ::startorch::ScalarType::kFloat64)                     \
  MACRO(std::float128_t, ::startorch::ScalarType::kFloat128)                   \
                                                                               \
  MACRO(std::bfloat16_t, ::startorch::ScalarType::kBrainFloat16)
#else
#define DARKSIDE_FORALL_CPP_TYPE_TO_SCALAR_TYPE(MACRO)                         \
  MACRO(std::monostate, ::startorch::ScalarType::kUndefined)                   \
  MACRO(bool, ::startorch::ScalarType::kBool)                                  \
                                                                               \
  MACRO(std::uint8_t, ::startorch::ScalarType::kUnsignedInt8)                  \
  MACRO(std::uint16_t, ::startorch::ScalarType::kUnsignedInt16)                \
  MACRO(std::uint32_t, ::startorch::ScalarType::kUnsignedInt32)                \
  MACRO(std::uint64_t, ::startorch::ScalarType::kUnsignedInt64)                \
                                                                               \
  MACRO(std::int8_t, ::startorch::ScalarType::kInt8)                           \
  MACRO(std::int16_t, ::startorch::ScalarType::kInt16)                         \
  MACRO(std::int32_t, ::startorch::ScalarType::kInt32)                         \
  MACRO(std::int64_t, ::startorch::ScalarType::kInt64)                         \
                                                                               \
  MACRO(float, ::startorch::ScalarType::kFloat32)                              \
  MACRO(double, ::startorch::ScalarType::kFloat64)
#endif

#endif // !DARKSIDE_MACROS_MAPPING_HPP_
