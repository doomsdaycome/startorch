#include "startorch/device/bridge.hpp"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstdint>
#include <cstring>

#include "startorch/common/types.hpp"
#include "startorch/device/engine.hpp"

namespace startorch {

Bridge::Bridge()
    : destination_engine_data_(nullptr),
      source_engine_data_(nullptr),
      type_(BridgeType::kUndefined) {}

Bridge::Bridge(Engine& destination_engine, const Engine& source_engine)
    : destination_engine_data_(&destination_engine),
      source_engine_data_(&source_engine),
      type_(BridgeType::kUndefined) {
  if (destination_engine_data_->IsNull() || source_engine_data_->IsNull()) {
    destination_engine_data_ = nullptr;
    source_engine_data_ = nullptr;
    return;
  }

  const EngineType destination_engine_type =
      destination_engine_data_->GetType();
  const EngineType source_engine_type = source_engine_data_->GetType();

  if (destination_engine_type == EngineType::kHost) {
    if (source_engine_type == EngineType::kHost)
      type_ = BridgeType::kHostToHost;
    else if (source_engine_type == EngineType::kDevice)
      type_ = BridgeType::kDeviceToHost;
  } else if (destination_engine_type == EngineType::kDevice) {
    if (source_engine_type == EngineType::kHost)
      type_ = BridgeType::kHostToDevice;
    else if (source_engine_type == EngineType::kDevice)
      type_ = BridgeType::kDeviceToDevice;
  }
}

Bridge::operator bool() const {
  return destination_engine_data_ && source_engine_data_ &&
         !destination_engine_data_->IsNull() && !source_engine_data_->IsNull();
}

Engine& Bridge::GetDestinationEngine() {
  if (destination_engine_data_ == nullptr) return Engine::GetNull();

  return *destination_engine_data_;
}

const Engine& Bridge::GetDestinationEngine() const {
  if (destination_engine_data_ == nullptr) return Engine::GetNull();

  return *destination_engine_data_;
}

const Engine& Bridge::GetSourceEngine() const { return *source_engine_data_; }
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
