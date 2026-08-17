#ifndef DARKSIDE_SCALAR_TRAITS_HPP_
#define DARKSIDE_SCALAR_TRAITS_HPP_

#include "darkside/common/types.hpp"
#include "darkside/macros/mapping.hpp"

namespace darkside {

template <ScalarType S>
struct ScalarTypeTraits;

#define DARKSIDE_DEF_SCALAR_TRAIT(S, C) \
  template <>                           \
  struct ScalarTypeTraits<S> {          \
    using type = C;                     \
  };

DARKSIDE_FORALL_SCALAR_CPP_TYPE(DARKSIDE_DEF_SCALAR_TRAIT)

#undef DARKSIDE_DEF_SCALAR_TRAIT

template <typename C>
struct CppTypeTraits;

#define DARKSIDE_DEF_CPP_TRAIT(C, S)       \
  template <>                              \
  struct CppTypeTraits<C> {                \
    static constexpr ScalarType value = S; \
  };

DARKSIDE_FORALL_CPP_SCALAR_TYPE(DARKSIDE_DEF_CPP_TRAIT)

#undef DARKSIDE_DEF_CPP_TRAIT

template <ScalarType S>
using scalar_t = typename ScalarTypeTraits<S>::type;

template <typename C>
inline constexpr ScalarType cpp_v = CppTypeTraits<C>::value;

}  // namespace darkside

#endif  // !DARKSIDE_SCALAR_TRAITS_HPP_
