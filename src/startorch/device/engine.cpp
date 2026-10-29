#include "startorch/device/engine.hpp"

#include <cstdint>

#include "startorch/common/types.hpp"
#include "startorch/device/memory.hpp"

namespace startorch {

Engine::Engine() : index_(0ul), memory_() {}

Engine::Engine(std::uint64_t index, std::uint64_t default_bytes,
               std::uint64_t virtual_bytes, EngineType system_type)
    : index_(index), memory_(default_bytes, virtual_bytes, system_type) {}

Engine::operator bool() const { return !memory_.IsNull(); }

std::uint64_t Engine::GetIndex() const { return index_; }

Memory& Engine::GetMemory() { return memory_; }

const Memory& Engine::GetMemory() const { return memory_; }

EngineType Engine::GetType() const { return memory_.GetEngineType(); }

bool Engine::IsNull() const { return !(*this); }

Engine& Engine::GetNull() {
  static Engine null_instance;
  return null_instance;
}

}  // namespace startorch
