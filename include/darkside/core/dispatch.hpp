#ifndef DARKSIDE_CORE_DISPATCH_HPP_
#define DARKSIDE_CORE_DISPATCH_HPP_

#include <stdexcept>
#include <utility>

#include "darkside/core/types.hpp"
#include "darkside/macros/mapping.hpp"

namespace darkside {

template <typename Fn>
decltype(auto) DispatchScalarType(ScalarType scalar_type, Fn&& fn) {
  switch (scalar_type) {
#define DARKSIDE_DISPATCH_CASE(S, C) \
  case S:                            \
    return std::forward<Fn>(fn).template operator()<C>();

    DARKSIDE_FORALL_SCALAR_CPP_TYPE(DARKSIDE_DISPATCH_CASE)

#undef DARKSIDE_DISPATCH_CASE

    default:
      throw std::invalid_argument(
          "darkside::DispatchScalarType: Undefined ScalarType.");
  }
}

#define DARKSIDE_DISPATCH_SCALAR_TYPE(S, ...) \
  DispatchScalarType(S, [&]<typename scalar_t>() { __VA_ARGS__; })

}  // namespace darkside

#endif  // !DARKSIDE_CORE_DISPATCH_HPP_
