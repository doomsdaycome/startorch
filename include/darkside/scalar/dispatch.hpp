#ifndef DARKSIDE_SCALAR_DISPATCH_HPP_
#define DARKSIDE_SCALAR_DISPATCH_HPP_

#include "darkside/macros/mapping.hpp"
#include "startorch/common/types.hpp"

#define DARKSIDE_DISPATCH_SCALAR_TYPE_CASE(S, C)                               \
  case S:                                                                      \
    return std::forward<Fn>(fn).template operator()<C>();

#define DARKSIDE_DISPATCH_SCALAR_TYPE(S, C, ...)                               \
  ::darkside::DispatchScalarType(                                              \
      S, [&]<typename C>() -> decltype(auto) { __VA_ARGS__; })

namespace darkside {

template <typename Fn>
decltype(auto) DispatchScalarType(startorch::ScalarType scalar_type, Fn &&fn) {
  switch (scalar_type) {
    DARKSIDE_FORALL_SCALAR_TYPE_TO_CPP_TYPE(DARKSIDE_DISPATCH_SCALAR_TYPE_CASE)
  default:
  }
}

} // namespace darkside

#undef DARKSIDE_DISPATCH_SCALAR_TYPE_CASE

#endif // !DARKSIDE_SCALAR_DISPATCH_HPP_
