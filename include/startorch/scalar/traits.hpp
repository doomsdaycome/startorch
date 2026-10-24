#ifndef STARTORCH_SCALAR_TRAITS_HPP_
#define STARTORCH_SCALAR_TRAITS_HPP_

#include "startorch/common/types.hpp"
#include "startorch/macros/mapping.hpp"

#define STARTORCH_DEF_CPP_TYPE_TRAITS(C, S) \
  template <>                               \
  struct CppTypeTraits<C> {                 \
    static constexpr ScalarType value = S;  \
  };

#define STARTORCH_DEF_SCALAR_TYPE_TRAITS(S, C) \
  template <>                                  \
  struct ScalarTypeTraits<S> {                 \
    using type = C;                            \
  };

namespace startorch {

template <typename C>
struct CppTypeTraits;

template <ScalarType S>
struct ScalarTypeTraits;

STARTORCH_FORALL_CPP_TYPE_TO_SCALAR_TYPE(STARTORCH_DEF_CPP_TYPE_TRAITS)
STARTORCH_FORALL_SCALAR_TYPE_TO_CPP_TYPE(STARTORCH_DEF_SCALAR_TYPE_TRAITS)

}  // namespace startorch

#undef STARTORCH_DEF_CPP_TYPE_TRAITS
#undef STARTORCH_DEF_SCALAR_TYPE_TRAITS

#endif  // !STARTORCH_SCALAR_TRAITS_HPP_
