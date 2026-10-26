#include "startorch/engine/system.hpp"

#include <cstdint>

#include "startorch/common/types.hpp"
#include "startorch/engine/memory.hpp"
#include "startorch/memory/buffer.hpp"

namespace startorch {

System::System() : index_(0ul), memory_() {}

System::System(std::uint64_t index, std::uint64_t physical_bytes,
               std::uint64_t virtual_bytes, SystemType system_type)
    : index_(index), memory_(physical_bytes, virtual_bytes, system_type) {}

System::operator bool() const { return !memory_.IsNull(); }

std::uint64_t System::GetIndex() const { return index_; }

Memory& System::GetMemory() { return memory_; }

const Memory& System::GetMemory() const { return memory_; }

SystemType System::GetType() const { return memory_.GetSystemType(); }

Buffer System::NewBuffer(std::uint64_t bytes, MallocType malloc_type) {
  if (bytes == 0ul || malloc_type == MallocType::kUndefined ||
      malloc_type == MallocType::kOptionCount)
    return Buffer::GetNull();

  switch (malloc_type) {
    case MallocType::kHost:
      if (memory_.GetSystemType() != SystemType::kHost) break;
      return memory_.GetPhysicalAllocator().NewBuffer(bytes);

    case MallocType::kPinned:
      if (memory_.GetSystemType() != SystemType::kHost) break;
      return memory_.GetVirtualAllocator().NewBuffer(bytes);

    case MallocType::kDevice:
      if (memory_.GetSystemType() != SystemType::kDevice) break;
      return memory_.GetPhysicalAllocator().NewBuffer(bytes);

    case MallocType::kUnified:
      if (memory_.GetSystemType() != SystemType::kDevice) break;
      return memory_.GetVirtualAllocator().NewBuffer(bytes);

    default:
      break;
  }

  return Buffer::GetNull();
}

void System::DeleteBuffer(Buffer& buffer) {
  if (buffer.IsNull()) return;

  switch (buffer.GetMallocType()) {
    case MallocType::kHost:
      if (memory_.GetSystemType() != SystemType::kHost) break;
      memory_.GetPhysicalAllocator().DeleteBuffer(buffer);
      break;

    case MallocType::kPinned:
      if (memory_.GetSystemType() != SystemType::kHost) break;
      memory_.GetVirtualAllocator().DeleteBuffer(buffer);
      break;

    case MallocType::kDevice:
      if (memory_.GetSystemType() != SystemType::kDevice) break;
      memory_.GetPhysicalAllocator().DeleteBuffer(buffer);
      break;

    case MallocType::kUnified:
      if (memory_.GetSystemType() != SystemType::kDevice) break;
      memory_.GetVirtualAllocator().DeleteBuffer(buffer);
      break;

    default:
      break;
  }
}

bool System::IsNull() const { return !(*this); }

System& System::GetNull() {
  static System null_instance;
  return null_instance;
}

}  // namespace startorch
