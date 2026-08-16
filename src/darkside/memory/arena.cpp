#include "darkside/memory/arena.hpp"

#include <cstdint>

#include "darkside/memory/pointer.hpp"

namespace darkside {

Arena::Arena(const Pointer& pointer, uint64_t size,
             Allocator* allocator_pointer)
    : pointer_(pointer), size_(size), allocator_pointer_(allocator_pointer) {}

}  // namespace darkside
