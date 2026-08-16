#ifndef DARKSIDE_ALLOCATOR_HPP_
#define DARKSIDE_ALLOCATOR_HPP_

#include <cstdint>
#include <memory>

#include "darkside/core/type.hpp"
#include "darkside/memory/pointer.hpp"
#include "darkside/memory/storage.hpp"

namespace darkside {

class Platform;

class Allocator {
 public:
  Allocator() = default;
  Allocator(const Allocator& other) = default;
  Allocator(Allocator&& other) noexcept = default;

  ~Allocator() = default;

  Allocator& operator=(const Allocator& other) = default;
  Allocator& operator=(Allocator&& other) noexcept = default;

 private:
  Storage storage_ = Storage();
  uint64_t offset_ = 0ul;
  std::weak_ptr<Platform> weak_platform_ = {};
};

class HostAllocator : public Allocator {};

class PinnedAllocator : public Allocator {};

class DeviceAllocator : public Allocator {};

class UnifiedAllocator : public Allocator {};

}  // namespace darkside

#endif  // !DARKSIDE_ALLOCATOR_HPP_
