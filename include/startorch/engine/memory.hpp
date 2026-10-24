#ifndef STARTORCH_ENGINE_MEMORY_HPP_
#define STARTORCH_ENGINE_MEMORY_HPP_

#include <cstdint>

#include "startorch/common/types.hpp"
#include "startorch/memory/allocator.hpp"

namespace startorch {

class Memory {
 public:
  Memory() = default;
  Memory(Memory&& other) noexcept = delete;
  Memory(const Memory& other) = delete;
  Memory(std::uint64_t physical_bytes, std::uint64_t virtual_bytes,
         SystemType system_type);

  ~Memory() = default;

  Memory& operator=(Memory&& other) noexcept = delete;
  Memory& operator=(const Memory& other) = delete;

  explicit operator bool() const;

  Allocator& GetPhysicalAllocator();
  const Allocator& GetPhysicalAllocator() const;
  Allocator& GetVirtualAllocator();
  const Allocator& GetVirtualAllocator() const;
  SystemType GetSystemType() const;

  bool IsNull() const;

  static Memory& GetNull();

 private:
  Allocator physical_allocator_;
  Allocator virtual_allocator_;
  SystemType system_type_ = SystemType::kUndefined;
};

}  // namespace startorch

#endif  // !STARTORCH_ENGINE_MEMORY_HPP_
