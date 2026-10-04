#ifndef STARTORCH_ENGINE_SYSTEM_HPP_
#define STARTORCH_ENGINE_SYSTEM_HPP_

#include "startorch/common/types.hpp"
#include "startorch/engine/memory.hpp"

namespace startorch {

class System {
public:
  System() = default;
  System(System &&other) noexcept = default;
  System(const System &other) = delete;
  System(std::uint64_t physical_bytes, std::uint64_t virtual_bytes,
         SystemType system_type);

  ~System() = default;

  System &operator=(System &&other) noexcept = default;
  System &operator=(const System &other) = delete;

  Memory &GetMemory();
  const Memory &GetMemory() const;
  SystemType GetType() const;

private:
  Memory memory_ = Memory();
  SystemType type_ = SystemType::kUndefined;
};

} // namespace startorch

#endif // !STARTORCH_ENGINE_SYSTEM_HPP_
