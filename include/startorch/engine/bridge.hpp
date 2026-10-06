#ifndef STARTORCH_ENGINE_BRIDGE_HPP_
#define STARTORCH_ENGINE_BRIDGE_HPP_

#include "startorch/common/types.hpp"
#include "startorch/engine/system.hpp"

namespace startorch {

class Bridge {
public:
  Bridge() = default;
  Bridge(Bridge &&other) noexcept = default;
  Bridge(const Bridge &other) = default;
  Bridge(const System &destination_system, const System &source_system);

  ~Bridge() = default;

  Bridge &operator=(Bridge &&other) noexcept = default;
  Bridge &operator=(const Bridge &other) = default;

  explicit operator bool() const;
  bool operator!() const;

  System &GetDestinationSystem();
  const System &GetDestinationSystem() const;
  System &GetSourceSystem();
  const System &GetSourceSystem() const;
  BridgeType GetType() const;

  bool IsNull() const;

  void Copy(darkside::Buffer &destination_buffer,
            const darkside::Buffer &source_buffer);

private:
  System *destination_system_data_ = nullptr;
  System *source_system_data_ = nullptr;
  BridgeType type_ = BridgeType::kUndefined;
};

} // namespace startorch

#endif // !STARTORCH_ENGINE_BRIDGE_HPP_
