#ifndef DARKSIDE_ALLOCATOR_HPP_
#define DARKSIDE_ALLOCATOR_HPP_

#include <cstdint>

#include "darkside/common/types.hpp"
#include "darkside/memory/arena.hpp"

namespace darkside {

class Memory;

class Allocator {
 public:
 private:
  Arena arena = Arena();
  uint64_t offset_ = 0ul;
  Memory* memory_pointer_ = nullptr;
};

class HostAllocator : public Allocator {};

class PinnedAllocator : public Allocator {};

class DeviceAllocator : public Allocator {};

class UnifiedAllocator : public Allocator {};

}  // namespace darkside

#endif  // !DARKSIDE_ALLOCATOR_HPP_
