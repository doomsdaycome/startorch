#ifndef DARKSIDE_SCALAR_TRAITS_HPP_
#define DARKSIDE_SCALAR_TRAITS_HPP_

#include "darkside/macros/mapping.hpp"
#include "startorch/common/types.hpp"

#define DARKSIDE_DEF_CPP_TYPE_TRAITS(C, S)                                     \
  template <> struct CppTypeTraits<C> {                                        \
    static constexpr startorch::ScalarType value = S;                          \
  };

#define DARKSIDE_DEF_SCALAR_TYPE_TRAITS(S, C)                                  \
  template <> struct ScalarTypeTraits<S> {                                     \
    using type = C;                                                            \
  };

namespace darkside {

template <typename C> struct CppTypeTraits;

template <startorch::ScalarType S> struct ScalarTypeTraits;

DARKSIDE_FORALL_CPP_TYPE_TO_SCALAR_TYPE(DARKSIDE_DEF_CPP_TYPE_TRAITS)
DARKSIDE_FORALL_SCALAR_TYPE_TO_CPP_TYPE(DARKSIDE_DEF_SCALAR_TYPE_TRAITS)

} // namespace darkside

#undef DARKSIDE_DEF_SCALAR_TYPE_TRAITS
#undef DARKSIDE_SCALAR_TRAITS_HPP_

#endif // !DARKSIDE_SCALAR_TRAITS_HPP_
