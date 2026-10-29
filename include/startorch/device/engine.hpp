#ifndef STARTORCH_DEVICE_ENGINE_HPP_
#define STARTORCH_DEVICE_ENGINE_HPP_

#include <cstdint>

#include "startorch/common/types.hpp"
#include "startorch/device/memory.hpp"

namespace startorch {

class Engine {
 public:
  Engine();
  Engine(Engine&& other) noexcept = delete;
  Engine(const Engine& other) = delete;
  Engine(std::uint64_t index, std::uint64_t physical_bytes,
         std::uint64_t virtual_bytes, EngineType system_type);

  ~Engine() = default;

  Engine& operator=(Engine&& other) noexcept = delete;
  Engine& operator=(const Engine& other) = delete;

  explicit operator bool() const;

  std::uint64_t GetIndex() const;
  Memory& GetMemory();
  const Memory& GetMemory() const;
  EngineType GetType() const;

  bool IsNull() const;

  static Engine& GetNull();

 private:
  std::uint64_t index_;
  Memory memory_;
};

}  // namespace startorch

#endif  // !STARTORCH_DEVICE_ENGINE_HPP_
