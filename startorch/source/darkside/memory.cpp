#include "darkside/memory.hpp"
#include "startorch/common.hpp"
#include "startorch/device.hpp"

#include <cstdint>
#include <cstring>
#include <new>

#include <cuda_runtime.h>

namespace darkside {
void *makeMemory(uint64_t size, const startorch::Device &device) {
  void *pointer = nullptr;

  if (device.getMemoryType() == startorch::MemoryType::PINNED) {
    if (cudaMallocHost(&pointer, size) != cudaSuccess)
      return nullptr;
    return pointer;
  }

  if (device.getMemoryType() == startorch::MemoryType::UNIFIED) {
    if (cudaMallocManaged(&pointer, size) != cudaSuccess)
      return nullptr;
    return pointer;
  }

  switch (device.getDeviceType()) {
  case startorch::DeviceType::CPU:
    pointer = new (std::nothrow) uint8_t[size];
    break;

  case startorch::DeviceType::GPU:
    if (cudaMalloc(&pointer, size) != cudaSuccess)
      pointer = nullptr;
    break;

  default:
    break;
  }

  return pointer;
}

void freeMemory(void *pointer, const startorch::Device &device) {
  if (pointer == nullptr)
    return;

  if (device.getMemoryType() == startorch::MemoryType::PINNED) {
    cudaFreeHost(pointer);
    return;
  }

  if (device.getMemoryType() == startorch::MemoryType::UNIFIED) {
    cudaFree(pointer);
    return;
  }

  switch (device.getDeviceType()) {
  case startorch::DeviceType::CPU:
    delete[] static_cast<uint8_t *>(pointer);
    break;

  case startorch::DeviceType::GPU:
    cudaFree(pointer);
    break;

  default:
    break;
  }
}

void copyMemory(void *destination, void *source, uint64_t size,
                const startorch::DevicePair &device_pair) {
  if (destination == nullptr || source == nullptr || size == 0)
    return;

  startorch::DeviceType first_device_type =
      device_pair.getFirstDevice().getDeviceType();
  startorch::DeviceType second_device_type =
      device_pair.getSecondDevice().getDeviceType();

  if (first_device_type == startorch::DeviceType::CPU &&
      second_device_type == startorch::DeviceType::CPU)
    memcpy(destination, source, size);
  else
    cudaMemcpy(destination, source, size, cudaMemcpyDefault);
}

Storage::Storage(uint64_t size, startorch::ScalarType scalar_type,
                 const startorch::Device &device)
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
             startorch::DevicePair(device_, other.device_));
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

  copyMemory(data_, other.data_, bytes,
             startorch::DevicePair(device_, other.device_));

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
startorch::ScalarType Storage::getScalarType() const { return scalar_type_; }
const startorch::Device &Storage::getDevice() const { return device_; }
} // namespace darkside
