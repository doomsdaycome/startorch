#ifndef STARTORCH_SCALAR_DISPATCH_HPP_
#define STARTORCH_SCALAR_DISPATCH_HPP_

#include <utility>

#include "startorch/common/types.hpp"
#include "startorch/macros/mapping.hpp"

#define STARTORCH_DISPATCH_SCALAR_TYPE_CASE(S, C) \
  case S:                                         \
    return std::forward<Fn>(fn).template operator()<C>();

#define STARTORCH_DISPATCH_SCALAR_TYPE(S, C, ...) \
  ::startorch::DispatchScalarType(                \
      S, [&]<typename C>() -> decltype(auto) { __VA_ARGS__; })

namespace startorch {

template <typename Fn>
decltype(auto) DispatchScalarType(ScalarType scalar_type, Fn&& fn) {
  switch (scalar_type) {
    STARTORCH_FORALL_SCALAR_TYPE_TO_CPP_TYPE(
        STARTORCH_DISPATCH_SCALAR_TYPE_CASE)

    default:
      break;
  }

  std::unreachable();
}

}  // namespace startorch

#undef STARTORCH_DISPATCH_SCALAR_TYPE_CASE

#endif  // !STARTORCH_SCALAR_DISPATCH_HPP_
