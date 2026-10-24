#include "startorch/scalar/utilities.hpp"

#include <cstdint>
#include <type_traits>
#include <variant>

#include "startorch/common/types.hpp"
#include "startorch/scalar/dispatch.hpp"

namespace startorch {

std::uint64_t SizeOfScalarType(startorch::ScalarType scalar_type) {
  return STARTORCH_DISPATCH_SCALAR_TYPE(scalar_type, C, {
    if constexpr (std::is_same_v<C, std::monostate>)
      return 0ul;
    else
      return SizeOfCPPType(C);
  });
}

}  // namespace startorch
