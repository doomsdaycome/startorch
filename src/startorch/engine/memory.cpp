#include "startorch/engine/memory.hpp"

#include <cstdint>

#include "darkside/memory/allocator.hpp"
#include "startorch/common/types.hpp"

namespace startorch {

startorch::MallocType GetPhysicalMallocType(SystemType system_type) {
  switch (system_type) {
  case SystemType::kHost:
    return startorch::MallocType::kHost;

  case SystemType::kDevice:
    return startorch::MallocType::kDevice;

  default:
    return startorch::MallocType::kUndefined;
  }
}

startorch::MallocType GetVirtualMallocType(SystemType system_type) {
  switch (system_type) {
  case SystemType::kHost:
    return startorch::MallocType::kPinned;

  case SystemType::kDevice:
    return startorch::MallocType::kUnified;

  default:
    return startorch::MallocType::kUndefined;
  }
}

constexpr SystemType GetValidSystemType(SystemType system_type) noexcept {
  switch (system_type) {
  case SystemType::kHost:
  case SystemType::kDevice:
    return system_type;

  default:
    return SystemType::kUndefined;
  }
}

Memory::Memory(std::uint64_t physical_bytes, std::uint64_t virtual_bytes,
               SystemType system_type)
    : physical_allocator_(physical_bytes, GetPhysicalMallocType(system_type)),
      virtual_allocator_(virtual_bytes, GetVirtualMallocType(system_type)),
      system_type_(GetValidSystemType(system_type)) {}

Memory &Memory::NullMemory() {
  static Memory null_instance;
  return null_instance;
}

Memory::operator bool() const {
  return !physical_allocator_.IsNull() || !virtual_allocator_.IsNull();
}

darkside::Allocator &Memory::GetPhysicalAllocator() {
  return physical_allocator_;
}

const darkside::Allocator &Memory::GetPhysicalAllocator() const {
  return physical_allocator_;
}

darkside::Allocator &Memory::GetVirtualAllocator() {
  return virtual_allocator_;
}

const darkside::Allocator &Memory::GetVirtualAllocator() const {
  return virtual_allocator_;
}

SystemType Memory::GetSystemType() const { return system_type_; }

bool Memory::IsNull() const { return !(*this); }

} // namespace startorch
