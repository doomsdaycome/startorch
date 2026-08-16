#ifndef DARKSIDE_CORE_TYPE_TRAITS_HPP_
#define DARKSIDE_CORE_TYPE_TRAITS_HPP_

#include "darkside/core/types.hpp"
#include "darkside/macros/expansion.hpp"

namespace darkside {

template <ScalarType S>
struct ScalarTypeTrait;

#define DARKSIDE_DEF_SCALAR_TYPE_TRAIT(S, C) \
  template <>                                \
  struct ScalarTypeTrait<S> {                \
    using type = C;                          \
  };

DARKSIDE_FORALL_SCALAR_CPP_TYPE(DARKSIDE_DEF_SCALAR_TYPE_TRAIT)

#undef DARKSIDE_DEF_SCALAR_TYPE_TRAIT

template <typename C>
struct CppTypeTrait;

#define DARKSIDE_DEF_CPP_TYPE_TRAIT(C, S)  \
  template <>                              \
  struct CppTypeTrait<C> {                 \
    static constexpr ScalarType value = S; \
  };

DARKSIDE_FORALL_CPP_SCALAR_TYPE(DARKSIDE_DEF_CPP_TYPE_TRAIT)

#undef DARKSIDE_DEF_CPP_TYPE_TRAIT

template <ScalarType S>
using scalar_t = typename ScalarTypeTrait<S>::type;

template <typename C>
inline constexpr ScalarType cpp_v = CppTypeTrait<C>::value;

}  // namespace darkside

#endif  // DARKSIDE_CORE_TYPE_TRAITS_HPP_
