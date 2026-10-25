#include "startorch/engine/memory.hpp"

#include <cstdint>

#include "startorch/common/types.hpp"
#include "startorch/memory/allocator.hpp"

namespace startorch {

Memory::Memory()
    : physical_allocator_(),
      virtual_allocator_(),
      system_type_(SystemType::kUndefined) {}

MallocType GetPhysicalMallocType(SystemType system_type) {
  switch (system_type) {
    case SystemType::kHost:
      return MallocType::kHost;

    case SystemType::kDevice:
      return MallocType::kDevice;

    default:
      return MallocType::kUndefined;
  }
}

MallocType GetVirtualMallocType(SystemType system_type) {
  switch (system_type) {
    case SystemType::kHost:
      return MallocType::kPinned;

    case SystemType::kDevice:
      return MallocType::kUnified;

    default:
      return MallocType::kUndefined;
  }
}

SystemType GetValidSystemType(SystemType system_type) {
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

Memory::operator bool() const {
  return !physical_allocator_.IsNull() || !virtual_allocator_.IsNull();
}

Allocator& Memory::GetPhysicalAllocator() { return physical_allocator_; }

const Allocator& Memory::GetPhysicalAllocator() const {
  return physical_allocator_;
}

Allocator& Memory::GetVirtualAllocator() { return virtual_allocator_; }

const Allocator& Memory::GetVirtualAllocator() const {
  return virtual_allocator_;
}

SystemType Memory::GetSystemType() const { return system_type_; }

bool Memory::IsNull() const { return !(*this); }

Memory& Memory::GetNull() {
  static Memory null_instance;
  return null_instance;
}

}  // namespace startorch
