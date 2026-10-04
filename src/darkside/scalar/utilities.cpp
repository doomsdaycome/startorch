#include "darkside/scalar/utilities.hpp"

#include <cstdint>
#include <type_traits>
#include <variant>

#include "darkside/scalar/dispatch.hpp"
#include "startorch/common/types.hpp"

namespace darkside {

std::uint64_t SizeOfScalarType(startorch::ScalarType scalar_type) {
  return DARKSIDE_DISPATCH_SCALAR_TYPE(scalar_type, C, {
    if constexpr (std::is_same_v<C, std::monostate>)
      return 0ul;
    else
      return SizeOfCPPType(C);
  });
}

} // namespace darkside
