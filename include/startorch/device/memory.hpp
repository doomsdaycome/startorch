#ifndef STARTORCH_DEVICE_MEMORY_HPP_
#define STARTORCH_DEVICE_MEMORY_HPP_

#include <cstdint>

#include "startorch/common/types.hpp"
#include "startorch/memory/allocator.hpp"

namespace startorch {

class Memory {
 public:
  Memory();
  Memory(Memory&& other) noexcept = delete;
  Memory(const Memory& other) = delete;
  Memory(std::uint64_t default_bytes, std::uint64_t virtual_bytes,
         EngineType engine_type);

  ~Memory() = default;

  Memory& operator=(Memory&& other) noexcept = delete;
  Memory& operator=(const Memory& other) = delete;

  explicit operator bool() const;

  Allocator& GetDefaultAllocator();
  const Allocator& GetDefaultAllocator() const;
  Allocator& GetVirtualAllocator();
  const Allocator& GetVirtualAllocator() const;
  EngineType GetEngineType() const;

  bool IsNull() const;

  static Memory& GetNull();

 private:
  Allocator default_allocator_;
  Allocator virtual_allocator_;
  EngineType engine_type_;
};

}  // namespace startorch

#endif  // !STARTORCH_DEVICE_MEMORY_HPP_
