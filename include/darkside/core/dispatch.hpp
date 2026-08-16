#ifndef DARKSIDE_CORE_DISPATCH_HPP_
#define DARKSIDE_CORE_DISPATCH_HPP_

#include <stdexcept>
#include <utility>

#include "darkside/core/type.hpp"
#include "darkside/macro/macro.hpp"

namespace darkside {

template <typename Fn>
decltype(auto) DispatchScalarType(ScalarType scalar_type, Fn&& fn) {
  switch (scalar_type) {
#define DARKSIDE_DISPATCH_CASE(SCALAR_TYPE, CPP_TYPE) \
  case SCALAR_TYPE:                                   \
    return std::forward<Fn>(fn).template operator()<CPP_TYPE>();

    DARKSIDE_FORALL_SCALAR_CPP_TYPE(DARKSIDE_DISPATCH_CASE)

#undef DARKSIDE_DISPATCH_CASE

    default:
      throw std::invalid_argument(
          "darkside::DispatchScalarType: Undefined ScalarType.");
  }
}

#define DARKSIDE_DISPATCH_SCALAR_TYPE(TYPE, ...) \
  DispatchScalarType(TYPE, [&]<typename scalar_t>() { __VA_ARGS__; })

}  // namespace darkside

#endif  // DARKSIDE_CORE_DISPATCH_HPP_
