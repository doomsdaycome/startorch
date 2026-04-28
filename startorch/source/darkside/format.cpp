#include "startorch/common.hpp"

#include "darkside/format.hpp"

#include <cstdint>

namespace darkside {
uint64_t getScalarTypeSize(startorch::ScalarType scalar_type) {
  switch (scalar_type) {
  case startorch::ScalarType::INT_8:
  case startorch::ScalarType::UNSIGNED_INT_8:
    return 1;

  case startorch::ScalarType::INT_16:
  case startorch::ScalarType::UNSIGNED_INT_16:
    return 2;

  case startorch::ScalarType::INT_32:
  case startorch::ScalarType::UNSIGNED_INT_32:
    return 4;

  case startorch::ScalarType::INT_64:
  case startorch::ScalarType::UNSIGNED_INT_64:
    return 8;

  case startorch::ScalarType::FLOAT_32:
    return sizeof(float);

  case startorch::ScalarType::FLOAT_64:
    return sizeof(double);

  default:
    return 0;
  }
}
} // namespace startorch
