#include "startorch/engine/system.hpp"

#include "startorch/common/types.hpp"
#include "startorch/engine/memory.hpp"

namespace startorch {

System::System(std::uint64_t physical_bytes, std::uint64_t virtual_bytes,
               SystemType system_type)
    : memory_(physical_bytes, virtual_bytes, system_type), type_(system_type) {}

System::operator bool() const {
  return memory_ && type_ != SystemType::kUndefined;
}

bool System::operator!() const {
  return !memory_ || type_ == SystemType::kUndefined;
}

Memory &System::GetMemory() { return memory_; }
const Memory &System::GetMemory() const { return memory_; }
SystemType System::GetType() const { return type_; }

} // namespace startorch
