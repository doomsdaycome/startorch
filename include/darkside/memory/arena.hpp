#ifndef DARKSIDE_MEMORY_ARENA_HPP_
#define DARKSIDE_MEMORY_ARENA_HPP_

#include <cstdint>

#include "darkside/memory/pointer.hpp"

namespace darkside {

class Allocator;

class Arena {
 public:
 private:
  Pointer pointer_ = Pointer();
  uint64_t size_ = 0ul;
  Allocator* allocator_raw_pointer_ = nullptr;
};

}  // namespace darkside

#endif  // !DARKSIDE_MEMORY_ARENA_HPP_
