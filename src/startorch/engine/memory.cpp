#include "startorch/engine/memory.hpp"

#include <cstdint>

#include "darkside/memory/allocator.hpp"
#include "startorch/common/types.hpp"

namespace startorch {

Memory::Memory(std::uint64_t physical_bytes, std::uint64_t virtual_bytes,
               SystemType system_type)
    : system_type_(system_type) {
  switch (system_type_) {
  case SystemType::kHost:
    physical_allocator_ =
        darkside::Allocator(physical_bytes, MallocType::kHost);
    virtual_allocator_ =
        darkside::Allocator(virtual_bytes, MallocType::kPinned);
    break;

  case SystemType::kDevice:
    physical_allocator_ =
        darkside::Allocator(physical_bytes, MallocType::kHost);
    virtual_allocator_ =
        darkside::Allocator(virtual_bytes, MallocType::kPinned);
    break;

  case SystemType::kUndefined:
  case SystemType::kOptionCount:
    system_type_ = SystemType::kUndefined;
    return;
  }
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
