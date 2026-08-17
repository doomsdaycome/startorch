#ifndef DARKSIDE_ALLOCATOR_HPP_
#define DARKSIDE_ALLOCATOR_HPP_

#include <cstdint>

#include "darkside/common/types.hpp"
#include "darkside/memory/arena.hpp"

namespace darkside {

class Memory;

class Allocator {
 public:
  Allocator(const Allocator& other) = delete;
  Allocator(Allocator&& other) noexcept;

  virtual ~Allocator() = 0;

  Allocator& operator=(const Allocator& other) = delete;
  Allocator& operator=(Allocator&& other) noexcept;

  Arena NewArena(ScalarType scalar_type, uint64_t size);
  void DeleteArena(const Arena& arena);

 protected:
  Allocator() = default;
  Allocator(const Arena& arena, uint64_t offset, Memory* memory_pointer);

  Arena arena_ = Arena();
  uint64_t offset_ = 0ul;
  Memory* memory_pointer_ = nullptr;
};

class HostAllocator : public Allocator {
 public:
  HostAllocator() = default;
  HostAllocator(const HostAllocator& other) = delete;
  HostAllocator(HostAllocator&& other) noexcept;

  HostAllocator(uint64_t bytes, Memory* memory_pointer);
  HostAllocator(const Arena& arena, uint64_t offset, Memory* memory_pointer);

  virtual ~HostAllocator() override;

  HostAllocator& operator=(const HostAllocator& other) = delete;
  HostAllocator& operator=(HostAllocator&& other) noexcept;
};

class PinnedAllocator : public Allocator {
 public:
  PinnedAllocator() = default;
  PinnedAllocator(const PinnedAllocator& other) = delete;
  PinnedAllocator(PinnedAllocator&& other) noexcept;

  PinnedAllocator(uint64_t bytes, Memory* memory_pointer);

  virtual ~PinnedAllocator() override;

  PinnedAllocator& operator=(const PinnedAllocator& other) = delete;
  PinnedAllocator& operator=(PinnedAllocator&& other) noexcept;
};

class DeviceAllocator : public Allocator {
 public:
  DeviceAllocator() = default;
  DeviceAllocator(const DeviceAllocator& other) = delete;
  DeviceAllocator(DeviceAllocator&& other) noexcept;

  DeviceAllocator(uint64_t bytes, Memory* memory_pointer);

  virtual ~DeviceAllocator() override;

  DeviceAllocator& operator=(const DeviceAllocator& other) = delete;
  DeviceAllocator& operator=(DeviceAllocator&& other) noexcept;
};

class UnifiedAllocator : public Allocator {
 public:
  UnifiedAllocator() = default;
  UnifiedAllocator(const UnifiedAllocator& other) = delete;
  UnifiedAllocator(UnifiedAllocator&& other) noexcept;

  UnifiedAllocator(uint64_t bytes, Memory* memory_pointer);

  virtual ~UnifiedAllocator() override;

  UnifiedAllocator& operator=(const UnifiedAllocator& other) = delete;
  UnifiedAllocator& operator=(UnifiedAllocator&& other) noexcept;
};

}  // namespace darkside

#endif  // !DARKSIDE_ALLOCATOR_HPP_
