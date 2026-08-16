#ifndef DARKSIDE_MEMORY_ARENA_HPP_
#define DARKSIDE_MEMORY_ARENA_HPP_

#include <cstdint>

#include "darkside/memory/pointer.hpp"

namespace darkside {

class Allocator;

class Arena {
 public:
  Arena() = default;
  Arena(const Arena& other) = default;
  Arena(Arena&& other) noexcept = default;

  Arena(const Pointer& pointer, uint64_t size, Allocator* allocator_pointer_);

  ~Arena() = default;

  Arena& operator=(const Arena& other) = default;
  Arena& operator=(Arena&& other) noexcept = default;

 private:
  Pointer pointer_ = Pointer();
  uint64_t size_ = 0ul;
  Allocator* allocator_pointer_ = nullptr;
};

}  // namespace darkside

#endif  // !DARKSIDE_MEMORY_ARENA_HPP_
