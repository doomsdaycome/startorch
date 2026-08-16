#ifndef DARKSIDE_ALLOCATOR_HPP_
#define DARKSIDE_ALLOCATOR_HPP_

#include <cstdint>

#include "darkside/core/types.hpp"
#include "darkside/memory/arena.hpp"

namespace darkside {

class Memory;

class Allocator {
 public:
  Allocator() = delete;
  Allocator(const Allocator& other) = delete;
  Allocator(Allocator&& other) noexcept = delete;

  virtual ~Allocator();

  Allocator& operator=(const Allocator& other) = delete;
  Allocator& operator=(Allocator&& other) noexcept = delete;

  Arena NewArena(ScalarType scalar_type, uint64_t size);
  void DeleteArena(Arena& arena);

 protected:
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
