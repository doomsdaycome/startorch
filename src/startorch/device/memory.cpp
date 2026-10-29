#include "startorch/device/memory.hpp"

#include <cstdint>

#include "startorch/common/types.hpp"
#include "startorch/memory/allocator.hpp"

namespace startorch {

Memory::Memory()
    : default_allocator_(),
      virtual_allocator_(),
      engine_type_(EngineType::kUndefined) {}

MallocType GetDefaultMallocType(EngineType system_type) {
  switch (system_type) {
    case EngineType::kHost:
      return MallocType::kHost;

    case EngineType::kDevice:
      return MallocType::kDevice;

    default:
      return MallocType::kUndefined;
  }
}

MallocType GetVirtualMallocType(EngineType system_type) {
  switch (system_type) {
    case EngineType::kHost:
      return MallocType::kPinned;

    case EngineType::kDevice:
      return MallocType::kUnified;

    default:
      return MallocType::kUndefined;
  }
}

EngineType GetValidEngineType(EngineType system_type) {
  switch (system_type) {
    case EngineType::kHost:
    case EngineType::kDevice:
      return system_type;

    default:
      return EngineType::kUndefined;
  }
}

Memory::Memory(std::uint64_t default_bytes, std::uint64_t virtual_bytes,
               EngineType system_type)
    : default_allocator_(default_bytes, GetDefaultMallocType(system_type)),
      virtual_allocator_(virtual_bytes, GetVirtualMallocType(system_type)),
      engine_type_(GetValidEngineType(system_type)) {}

Memory::operator bool() const {
  return !default_allocator_.IsNull() || !virtual_allocator_.IsNull();
}

Allocator& Memory::GetDefaultAllocator() { return default_allocator_; }

const Allocator& Memory::GetDefaultAllocator() const {
  return default_allocator_;
}

Allocator& Memory::GetVirtualAllocator() { return virtual_allocator_; }

const Allocator& Memory::GetVirtualAllocator() const {
  return virtual_allocator_;
}

EngineType Memory::GetEngineType() const { return engine_type_; }

bool Memory::IsNull() const { return !(*this); }

Memory& Memory::GetNull() {
  static Memory null_instance;
  return null_instance;
}

}  // namespace startorch
