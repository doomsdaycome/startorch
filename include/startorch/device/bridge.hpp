#ifndef STARTORCH_ENGINE_BRIDGE_HPP_
#define STARTORCH_ENGINE_BRIDGE_HPP_

#include "startorch/common/types.hpp"
#include "startorch/device/engine.hpp"

namespace startorch {

class Bridge {
 public:
  Bridge();
  Bridge(Bridge&& other) noexcept = default;
  Bridge(const Bridge& other) = default;
  Bridge(Engine& destination_engine, const Engine& source_engine);

  ~Bridge() = default;

  Bridge& operator=(Bridge&& other) noexcept = default;
  Bridge& operator=(const Bridge& other) = default;

  explicit operator bool() const;

  Engine& GetDestinationEngine();
  const Engine& GetDestinationEngine() const;
  Engine& GetSourceEngine();
  const Engine& GetSourceEngine() const;
  BridgeType GetType() const;

  bool IsNull() const;

  void Copy(Buffer& destination_buffer, const Buffer& source_buffer);

  static Bridge& GetNull();

 private:
  Engine* destination_engine_data_;
  const Engine* source_engine_data_;
  BridgeType type_;
};

}  // namespace startorch

#endif  // !STARTORCH_ENGINE_BRIDGE_HPP_
