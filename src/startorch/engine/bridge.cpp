#include "startorch/engine/bridge.hpp"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>

#include <cuda_runtime_api.h>

#include "startorch/common/types.hpp"
#include "startorch/engine/system.hpp"

namespace startorch {

Bridge::Bridge(System &destination_system, const System &source_system)
    : destination_system_data_(&destination_system),
      source_system_data_(&source_system), type_(BridgeType::kUndefined) {
  if (destination_system_data_->IsNull() || source_system_data_->IsNull()) {
    destination_system_data_ = nullptr;
    source_system_data_ = nullptr;
    return;
  }

  const SystemType destination_system_type =
      destination_system_data_->GetType();
  const SystemType source_system_type = source_system_data_->GetType();

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
  return destination_system_data_ && source_system_data_ &&
         !destination_system_data_->IsNull() && !source_system_data_->IsNull();
}

System &Bridge::GetDestinationSystem() {
  if (destination_system_data_ == nullptr)
    return System::GetNull();

  return *destination_system_data_;
}

const System &Bridge::GetDestinationSystem() const {
  if (destination_system_data_ == nullptr)
    return System::GetNull();

  return *destination_system_data_;
}

const System &Bridge::GetSourceSystem() const { return *source_system_data_; }
BridgeType Bridge::GetType() const { return type_; }

bool Bridge::IsNull() const { return !(*this); }

void Bridge::Copy(darkside::Buffer &destination_buffer,
                  const darkside::Buffer &source_buffer) {
  if (destination_buffer.IsNull() || source_buffer.IsNull())
    return;

  std::uint64_t bytes =
      std::min(destination_buffer.GetBytes(), source_buffer.GetBytes());

  if (bytes == 0)
    return;

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
    break;
  }
}

} // namespace startorch
