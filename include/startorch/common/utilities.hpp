#ifndef STARTORCH_COMMON_UTILITIES_HPP_
#define STARTORCH_COMMON_UTILITIES_HPP_

#include <cstdint>

#include "startorch/common/types.hpp"

#define SizeOfCPPType(C) sizeof(C)

namespace startorch {

std::uint64_t SizeOfScalarType(ScalarType scalar_type);

}

#endif  // !STARTORCH_COMMON_UTILITIES_HPP_
