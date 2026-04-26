#include "startorch/common.hpp"
#include <cstdint>
namespace darkside {
template <typename T> struct CppToScalar;
#define CPP_TO_SCALAR(cpp_type, scalar_type)                                   \
  template <> struct CppToScalar<cpp_type> {                                   \
    static constexpr startorch::ScalarType type = scalar_type;                 \
  };
CPP_TO_SCALAR(int8_t, startorch::ScalarType::INT_8);
CPP_TO_SCALAR(int16_t, startorch::ScalarType::INT_16);
CPP_TO_SCALAR(int32_t, startorch::ScalarType::INT_32);
CPP_TO_SCALAR(int64_t, startorch::ScalarType::INT_64);
CPP_TO_SCALAR(float, startorch::ScalarType::FLOAT_32);
CPP_TO_SCALAR(double, startorch::ScalarType::INT_64);
CPP_TO_SCALAR(uint8_t, startorch::ScalarType::UNSIGNED_INT_8);
CPP_TO_SCALAR(uint16_t, startorch::ScalarType::UNSIGNED_INT_16);
CPP_TO_SCALAR(uint32_t, startorch::ScalarType::UNSIGNED_INT_32);
CPP_TO_SCALAR(uint64_t, startorch::ScalarType::UNSIGNED_INT_64);
#undef CPP_TO_SCALAR
template <startorch::ScalarType S> struct ScalarToCpp;
#define SCALAR_TO_CPP(scalar_type, cpp_type)                                   \
  template <> struct ScalarToCpp<scalar_type> {                                \
    using type = cpp_type;                                                     \
  };
SCALAR_TO_CPP(startorch::ScalarType::INT_8, int8_t);
SCALAR_TO_CPP(startorch::ScalarType::INT_16, int16_t);
SCALAR_TO_CPP(startorch::ScalarType::INT_32, int32_t);
SCALAR_TO_CPP(startorch::ScalarType::INT_64, int64_t);
SCALAR_TO_CPP(startorch::ScalarType::FLOAT_32, float);
SCALAR_TO_CPP(startorch::ScalarType::FLOAT_64, double);
SCALAR_TO_CPP(startorch::ScalarType::UNSIGNED_INT_8, uint8_t);
SCALAR_TO_CPP(startorch::ScalarType::UNSIGNED_INT_16, uint16_t);
SCALAR_TO_CPP(startorch::ScalarType::UNSIGNED_INT_32, uint32_t);
SCALAR_TO_CPP(startorch::ScalarType::UNSIGNED_INT_64, uint64_t);
#undef SCALAR_TO_CPP
} // namespace darkside
