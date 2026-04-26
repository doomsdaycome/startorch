#include "startorch/memory.hpp"
#include "startorch/common.hpp"
#include "startorch/device.hpp"

#include <cstdint>
#include <cstring>
#include <new>

#include <cuda_runtime.h>

namespace startorch {
void *makeMemory(uint64_t size, const Device &device) {
  void *pointer = nullptr;

  if (device.getMemoryType() == MemoryType::PINNED) {
    if (cudaMallocHost(&pointer, size) != cudaSuccess)
      return nullptr;
    return pointer;
  }

  if (device.getMemoryType() == MemoryType::UNIFIED) {
    if (cudaMallocManaged(&pointer, size) != cudaSuccess)
      return nullptr;
    return pointer;
  }

  switch (device.getDeviceType()) {
  case DeviceType::CPU:
    pointer = new (std::nothrow) uint8_t[size];
    break;

  case DeviceType::GPU:
    if (cudaMalloc(&pointer, size) != cudaSuccess)
      pointer = nullptr;
    break;

  default:
    break;
  }

  return pointer;
}

void freeMemory(void *pointer, const Device &device) {
  if (pointer == nullptr)
    return;

  if (device.getMemoryType() == MemoryType::PINNED) {
    cudaFreeHost(pointer);
    return;
  }

  if (device.getMemoryType() == MemoryType::UNIFIED) {
    cudaFree(pointer);
    return;
  }

  switch (device.getDeviceType()) {
  case DeviceType::CPU:
    delete[] static_cast<uint8_t *>(pointer);
    break;

  case DeviceType::GPU:
    cudaFree(pointer);
    break;

  default:
    break;
  }
}

void copyMemory(void *destination, void *source, uint64_t size,
                const DevicePair &device_pair) {
  if (destination == nullptr || source == nullptr || size == 0)
    return;

  DeviceType first_device_type = device_pair.getFirstDevice().getDeviceType();
  DeviceType second_device_type = device_pair.getSecondDevice().getDeviceType();

  if (first_device_type == DeviceType::CPU &&
      second_device_type == DeviceType::CPU)
    memcpy(destination, source, size);
  else
    cudaMemcpy(destination, source, size, cudaMemcpyDefault);
}

Storage::Storage(uint64_t size, ScalarType scalar_type, const Device &device)
    : data_(nullptr), size_(size), scalar_type_(scalar_type), device_(device) {
  if (size_ == 0)
    return;

  data_ = makeMemory(size_ * getScalarSize(scalar_type_), device_);

  if (data_ == nullptr)
    size_ = 0;
}

Storage::Storage(const Storage &other)
    : data_(nullptr), size_(other.size_), scalar_type_(other.scalar_type_),
      device_(other.device_) {
  if (size_ == 0)
    return;

  data_ = makeMemory(size_ * getScalarSize(scalar_type_), device_);

  if (data_ == nullptr) {
    size_ = 0;
    return;
  }

  copyMemory(data_, other.data_, size_ * getScalarSize(scalar_type_),
             DevicePair(device_, other.device_));
}

Storage::Storage(Storage &&other) noexcept
    : data_(other.data_), size_(other.size_), scalar_type_(other.scalar_type_),
      device_(other.device_) {
  other.data_ = nullptr;
  other.size_ = 0;
}

Storage::~Storage() {
  if (data_ != nullptr) {
    freeMemory(data_, device_);
  }
}

Storage &Storage::operator=(const Storage &other) {
  if (this == &other)
    return *this;

  if (other.size_ == 0) {
    if (size_ != 0)
      freeMemory(data_, device_);

    data_ = nullptr;
    size_ = 0;
    scalar_type_ = other.scalar_type_;
    device_ = other.device_;

    return *this;
  }

  uint64_t bytes = other.size_ * getScalarSize(other.scalar_type_);
  void *new_data = makeMemory(bytes, other.device_);

  if (new_data == nullptr)
    return *this;

  if (size_ != 0)
    freeMemory(data_, device_);

  data_ = new_data;
  size_ = other.size_;
  scalar_type_ = other.scalar_type_;
  device_ = other.device_;

  copyMemory(data_, other.data_, bytes, DevicePair(device_, other.device_));

  return *this;
}

Storage &Storage::operator=(Storage &&other) noexcept {
  if (this == &other)
    return *this;

  if (size_ != 0)
    freeMemory(data_, device_);

  data_ = other.data_;
  size_ = other.size_;
  scalar_type_ = other.scalar_type_;
  device_ = other.device_;

  other.data_ = nullptr;
  other.size_ = 0;

  return *this;
}

void *Storage::getData() const { return data_; }
uint64_t Storage::getSize() const { return size_; }
ScalarType Storage::getScalarType() const { return scalar_type_; }
const Device &Storage::getDevice() const { return device_; }

uint64_t getScalarSize(startorch::ScalarType scalar_type) {
  switch (scalar_type) {
  case ScalarType::INT_8:
  case ScalarType::UNSIGNED_INT_8:
    return 1;

  case ScalarType::INT_16:
  case ScalarType::UNSIGNED_INT_16:
    return 2;

  case ScalarType::INT_32:
  case ScalarType::UNSIGNED_INT_32:
    return 4;

  case ScalarType::INT_64:
  case ScalarType::UNSIGNED_INT_64:
    return 8;

  case ScalarType::FLOAT_32:
    return sizeof(float);

  case ScalarType::FLOAT_64:
    return sizeof(double);

  default:
    return 0;
  }
}
uint64_t getStorageSize(const Storage &storage) {
  return storage.getSize() * getScalarSize(storage.getScalarType());
}
} // namespace startorch
