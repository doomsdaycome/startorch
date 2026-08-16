#include "darkside/memory/pointer.hpp"

#include "darkside/core/type.hpp"
#include "darkside/memory/value.hpp"

namespace darkside {

Pointer::Pointer(void* pointer, ScalarType scalar_type, Arena* arena_pointer)
    : pointer_(pointer),
      scalar_type_(scalar_type),
      arena_pointer_(arena_pointer) {}

}  // namespace darkside
