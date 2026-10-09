#include "startorch/engine/system.hpp"

#include <cstdint>

#include "startorch/common/types.hpp"
#include "startorch/engine/memory.hpp"

namespace startorch {

System::System(std::uint64_t index, std::uint64_t physical_bytes,
               std::uint64_t virtual_bytes, SystemType system_type)
    : index_(index), memory_(physical_bytes, virtual_bytes, system_type) {}

System::operator bool() const { return !memory_.IsNull(); }

std::uint64_t System::GetIndex() const { return index_; }
Memory &System::GetMemory() { return memory_; }
const Memory &System::GetMemory() const { return memory_; }
SystemType System::GetType() const { return memory_.GetSystemType(); }
bool System::IsNull() const { return !(*this); }

System &System::GetNull() {
  static System null_instance;
  return null_instance;
}

} // namespace startorch
