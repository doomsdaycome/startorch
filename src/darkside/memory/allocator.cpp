#include "darkside/memory/allocator.hpp"

#include <cstdint>

#include "darkside/common/types.hpp"
#include "darkside/memory/arena.hpp"

namespace darkside {

Allocator::Allocator(Allocator&& other) noexcept
    : arena_(other.arena_),
      offset_(other.offset_),
      memory_pointer_(other.memory_pointer_) {
  other.arena_ = Arena();
  other.offset_ = 0ul;
  other.memory_pointer_ = nullptr;
}

Allocator& Allocator::operator=(Allocator&& other) noexcept {
  if (this != &other) {
    arena_ = other.arena_;
    offset_ = other.offset_;
    memory_pointer_ = other.memory_pointer_;

    other.arena_ = Arena();
    other.offset_ = 0ul;
    other.memory_pointer_ = nullptr;
  }

  return *this;
}

// Arena Allocator::NewArena(ScalarType scalar_type, uint64_t size) {}
//
// void Allocator::DeleteArena(const Arena& arena) {}
//
// Allocator::Allocator(const Arena& arena, uint64_t offset,
//                      Memory* memory_pointer) {}

}  // namespace darkside
