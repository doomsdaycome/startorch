#include "startorch/engine/bridge.hpp"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstdint>
#include <cstring>

#include "startorch/common/types.hpp"
#include "startorch/common/variables.hpp"
#include "startorch/engine/system.hpp"

namespace startorch {

Bridge::Bridge()
    : destination_system_(kDefaultHostSystem),
      source_system_(kDefaultHostSystem),
      type_(BridgeType::kHostToHost) {}

Bridge::Bridge(Bridge&& other) noexcept
    : destination_system_(other.destination_system_),
      source_system_(other.source_system_),
      type_(other.type_) {}

Bridge::Bridge(const Bridge& other)
    : destination_system_(other.destination_system_),
      source_system_(other.source_system_),
      type_(other.type_) {}

Bridge::Bridge(System& destination_system, const System& source_system)
    : destination_system_(destination_system),
      source_system_(source_system),
      type_(BridgeType::kUndefined) {
  if (destination_system_.IsNull() || source_system_.IsNull()) {
    return;
  }

  const SystemType destination_system_type = destination_system_.GetType();
  const SystemType source_system_type = source_system_.GetType();

  if (destination_system_type == SystemType::kHost) {
    if (source_system_type == SystemType::kHost)
      type_ = BridgeType::kHostToHost;
    else if (source_system_type == SystemType::kDevice)
      type_ = BridgeType::kDeviceToHost;
  } else if (destination_system_type == SystemType::kDevice) {
    if (source_system_type == SystemType::kHost)
      type_ = BridgeType::kHostToDevice;
    else if (source_system_type == SystemType::kDevice)
      type_ = BridgeType::kDeviceToDevice;
  }
}

Bridge::operator bool() const {
  return !destination_system_.IsNull() && !source_system_.IsNull();
}

System& Bridge::GetDestinationSystem() { return destination_system_; }

const System& Bridge::GetDestinationSystem() const {
  return destination_system_;
}

const System& Bridge::GetSourceSystem() const { return source_system_; }

BridgeType Bridge::GetType() const { return type_; }

bool Bridge::IsNull() const { return !(*this); }

void Bridge::Copy(Buffer& destination_buffer, const Buffer& source_buffer) {
  if (destination_buffer.IsNull() || source_buffer.IsNull()) return;

  std::uint64_t bytes =
      std::min(destination_buffer.GetBytes(), source_buffer.GetBytes());

  if (bytes == 0) return;

  switch (type_) {
    case BridgeType::kHostToHost:
      std::memcpy(destination_buffer.GetData(), source_buffer.GetData(), bytes);
      break;

    case BridgeType::kHostToDevice:
      cudaMemcpy(destination_buffer.GetData(), source_buffer.GetData(), bytes,
                 cudaMemcpyHostToDevice);
      break;

    case BridgeType::kDeviceToHost:
      cudaMemcpy(destination_buffer.GetData(), source_buffer.GetData(), bytes,
                 cudaMemcpyDeviceToHost);
      break;

    case BridgeType::kDeviceToDevice:
      cudaMemcpy(destination_buffer.GetData(), source_buffer.GetData(), bytes,
                 cudaMemcpyDeviceToDevice);
      break;

    default:
      cudaMemcpy(destination_buffer.GetData(), source_buffer.GetData(), bytes,
                 cudaMemcpyDefault);
      break;
  }
}

Bridge& Bridge::GetNull() {
  static Bridge null_instance;
  return null_instance;
}

}  // namespace startorch
