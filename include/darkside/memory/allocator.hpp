#ifndef DARKSIDE_ALLOCATOR_HPP_
#define DARKSIDE_ALLOCATOR_HPP_

#include <cstdint>

#include "darkside/common/types.hpp"
#include "darkside/memory/arena.hpp"

namespace darkside {

class Memory;

class Allocator {
 public:
  Allocator(Allocator&& other);
  Allocator(const Allocator& other) = delete;

  virtual ~Allocator() = 0;

  Allocator& operator=(Allocator&& other);
  Allocator& operator=(const Allocator& other) = delete;

  Arena NewArena(ScalarType scalar_type, uint64_t size);
  void DeleteArena(const Arena& arena);

 protected:
  Allocator() = default;
  Allocator(const Arena& arena, uint64_t offset, Memory* memory_pointer);

  Arena arena_ = Arena();
  uint64_t offset_ = 0ul;
  Memory* memory_raw_pointer_ = nullptr;
};

class HostAllocator : public Allocator {
 public:
  HostAllocator() = default;
  HostAllocator(HostAllocator&& other);
  HostAllocator(const HostAllocator& other) = delete;

  HostAllocator(uint64_t bytes, Memory* memory_pointer);
  HostAllocator(const Arena& arena, uint64_t offset, Memory* memory_pointer);

  virtual ~HostAllocator() override;

  HostAllocator& operator=(HostAllocator&& other);
  HostAllocator& operator=(const HostAllocator& other) = delete;
};

class PinnedAllocator : public Allocator {
 public:
  PinnedAllocator() = default;
  PinnedAllocator(PinnedAllocator&& other);
  PinnedAllocator(const PinnedAllocator& other) = delete;

  PinnedAllocator(uint64_t bytes, Memory* memory_pointer);

  virtual ~PinnedAllocator() override;

  PinnedAllocator& operator=(PinnedAllocator&& other);
  PinnedAllocator& operator=(const PinnedAllocator& other) = delete;
};

class DeviceAllocator : public Allocator {
 public:
  DeviceAllocator() = default;
  DeviceAllocator(DeviceAllocator&& other);
  DeviceAllocator(const DeviceAllocator& other) = delete;

  DeviceAllocator(uint64_t bytes, Memory* memory_pointer);

  virtual ~DeviceAllocator() override;

  DeviceAllocator& operator=(DeviceAllocator&& other);
  DeviceAllocator& operator=(const DeviceAllocator& other) = delete;
};

class UnifiedAllocator : public Allocator {
 public:
  UnifiedAllocator() = default;
  UnifiedAllocator(UnifiedAllocator&& other);
  UnifiedAllocator(const UnifiedAllocator& other) = delete;

  UnifiedAllocator(uint64_t bytes, Memory* memory_pointer);

  virtual ~UnifiedAllocator() override;

  UnifiedAllocator& operator=(UnifiedAllocator&& other);
  UnifiedAllocator& operator=(const UnifiedAllocator& other) = delete;
};

}  // namespace darkside

#endif  // !DARKSIDE_ALLOCATOR_HPP_
