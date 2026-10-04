#ifndef STARTORCH_ENGINE_MEMORY_HPP_
#define STARTORCH_ENGINE_MEMORY_HPP_

#include <cstdint>

#include "darkside/memory/allocator.hpp"
#include "startorch/common/types.hpp"

namespace startorch {

class Memory {
public:
  Memory() = default;
  Memory(Memory &&other) noexcept = default;
  Memory(const Memory &other) = delete;
  Memory(std::uint64_t physical_bytes, std::uint64_t virtual_bytes,
         SystemType system_type);

  ~Memory() = default;

  Memory &operator=(Memory &&other) noexcept = default;
  Memory &operator=(const Memory &other) = delete;

  explicit operator bool() const;
  bool operator!() const;

  darkside::Allocator &GetPhysicalAllocator();
  const darkside::Allocator &GetPhysicalAllocator() const;
  darkside::Allocator &GetVirtualAllocator();
  const darkside::Allocator &GetVirtualAllocator() const;

  bool IsNull() const;

private:
  darkside::Allocator physical_allocator_ = darkside::Allocator();
  darkside::Allocator virtual_allocator_ = darkside::Allocator();
  SystemType system_type_ = SystemType::kUndefined;
};

} // namespace startorch

#endif // !STARTORCH_ENGINE_MEMORY_HPP_
