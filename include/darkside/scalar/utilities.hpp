#ifndef DARKSIDE_SCALAR_UTILITIES_HPP_
#define DARKSIDE_SCALAR_UTILITIES_HPP_

#include <cstdint>

#include "startorch/common/types.hpp"

#define SizeOfCPPType(C) sizeof(C)

namespace darkside {

std::uint64_t SizeOfScalarType(startorch::ScalarType scalar_type);

}

#endif // !DARKSIDE_SCALAR_UTILITIES_HPP_
