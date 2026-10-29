#ifndef STARTORCH_DEVICE_SYSTEM_HPP_
#define STARTORCH_DEVICE_SYSTEM_HPP_

#include <cstdint>
#include <unordered_map>
#include <vector>

#include "startorch/common/types.hpp"
#include "startorch/device/engine.hpp"
#include "startorch/device/memory.hpp"
#include "startorch/memory/buffer.hpp"

namespace startorch {

class System {
 public:
  System() = default;
  System(System&& other) noexcept = default;
  System(const System& other) = default;

  ~System() = default;

  System& operator=(System&& other) noexcept = default;
  System& operator=(const System& other) = default;

  explicit operator bool() const;

  std::uint64_t GetIndex() const;
  Memory& GetMemory();
  const Memory& GetMemory() const;

  Buffer NewBuffer(std::uint64_t bytes, MallocType malloc_type);
  void DeleteBuffer(Buffer& buffer);

  bool IsNull() const;

  static System& GetNull();

 private:
  std::unordered_map<EngineType, std::vector<Engine&>> engine_map_;
};

}  // namespace startorch

#endif  // !STARTORCH_ENGINE_SYSTEM_HPP_
