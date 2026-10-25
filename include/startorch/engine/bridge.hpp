#ifndef STARTORCH_ENGINE_BRIDGE_HPP_
#define STARTORCH_ENGINE_BRIDGE_HPP_

#include "startorch/common/types.hpp"
#include "startorch/engine/system.hpp"

namespace startorch {

class Bridge {
 public:
  Bridge();
  Bridge(Bridge&& other) noexcept;
  Bridge(const Bridge& other);
  Bridge(System& destination_system, const System& source_system);

  ~Bridge() = default;

  Bridge& operator=(Bridge&& other) noexcept = delete;
  Bridge& operator=(const Bridge& other) = delete;

  explicit operator bool() const;

  System& GetDestinationSystem();
  const System& GetDestinationSystem() const;
  System& GetSourceSystem();
  const System& GetSourceSystem() const;
  BridgeType GetType() const;

  bool IsNull() const;

  void Copy(Buffer& destination_buffer, const Buffer& source_buffer);

  static Bridge& GetNull();

 private:
  System& destination_system_;
  const System& source_system_;
  BridgeType type_;
};

}  // namespace startorch

#endif  // !STARTORCH_ENGINE_BRIDGE_HPP_
