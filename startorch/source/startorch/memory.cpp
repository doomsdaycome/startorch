#include "startorch/memory.hpp"
#include "startorch/common.hpp"
#include "startorch/device.hpp"

#include "darkside/assign.hpp"
#include "darkside/format.hpp"
#include "darkside/memory.hpp"

#include <cstdint>

namespace startorch {
Storage::Storage(uint64_t size, ScalarType scalar_type, const Device &device)
    : size_(size), scalar_type_(scalar_type), device_(device) {
  if (size_ == 0)
    return;

  data_ = darkside::makeData(size_ * darkside::getScalarTypeSize(scalar_type_),
                             device_);

  if (data_ == nullptr)
    size_ = 0;
}

Storage::Storage(const Storage &other)
    : size_(other.size_), scalar_type_(other.scalar_type_),
      device_(other.device_) {
  if (size_ == 0)
    return;

  data_ = darkside::makeData(size_ * darkside::getScalarTypeSize(scalar_type_),
                             device_);

  if (data_ == nullptr) {
    size_ = 0;
    return;
  }

  darkside::copyData(data_, other.data_,
                     size_ * darkside::getScalarTypeSize(scalar_type_),
                     DevicePair(device_, other.device_));
}

Storage::Storage(Storage &&other) noexcept
    : data_(other.data_), size_(other.size_), scalar_type_(other.scalar_type_),
      device_(other.device_) {
  other.data_ = nullptr;
  other.size_ = 0;
}

Storage::~Storage() {
  if (data_ != nullptr)
    darkside::freeData(data_, device_);
}

Storage &Storage::operator=(const Storage &other) {
  if (this == &other)
    return *this;

  if (other.size_ == 0) {
    if (size_ != 0)
      darkside::freeData(data_, device_);

    data_ = nullptr;
    size_ = 0;
    scalar_type_ = other.scalar_type_;
    device_ = other.device_;

    return *this;
  }

  uint64_t bytes =
      other.size_ * darkside::getScalarTypeSize(other.scalar_type_);
  void *new_data = darkside::makeData(bytes, other.device_);

  if (new_data == nullptr)
    return *this;

  if (size_ != 0)
    darkside::freeData(data_, device_);

  data_ = new_data;
  size_ = other.size_;
  scalar_type_ = other.scalar_type_;
  device_ = other.device_;

  darkside::copyData(data_, other.data_, bytes,
                     DevicePair(device_, other.device_));

  return *this;
}

Storage &Storage::operator=(Storage &&other) noexcept {
  if (this == &other)
    return *this;

  if (size_ != 0)
    darkside::freeData(data_, device_);

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

void Storage::setDevice(const Device &device) {
  if (device == device_ || size_ == 0) {
    device_ = device;
    return;
  }

  uint64_t bytes = size_ * darkside::getScalarTypeSize(scalar_type_);
  void *new_data = darkside::makeData(bytes, device);

  if (new_data == nullptr)
    return;

  darkside::copyData(new_data, data_, bytes, DevicePair(device_, device));

  if (data_ != nullptr)
    darkside::freeData(data_, device_);

  data_ = new_data;
  device_ = device;
}

void Storage::fillData(const ScalarToCPP &value) {
  switch (scalar_type_) {
  case ScalarType::INT_8:
    darkside::fillData<int8_t>(data_, size_, value.value<int8_t>(), device_);
    break;

  case ScalarType::INT_16:
    darkside::fillData<int16_t>(data_, size_, value.value<int16_t>(), device_);
    break;

  case ScalarType::INT_32:
    darkside::fillData<int32_t>(data_, size_, value.value<int32_t>(), device_);
    break;

  case ScalarType::INT_64:
    darkside::fillData<int64_t>(data_, size_, value.value<int64_t>(), device_);
    break;

  case ScalarType::FLOAT_32:
    darkside::fillData<float>(data_, size_, value.value<float>(), device_);
    break;

  case ScalarType::FLOAT_64:
    darkside::fillData<double>(data_, size_, value.value<double>(), device_);
    break;

  case ScalarType::UNSIGNED_INT_8:
    darkside::fillData<uint8_t>(data_, size_, value.value<uint8_t>(), device_);
    break;

  case ScalarType::UNSIGNED_INT_16:
    darkside::fillData<uint16_t>(data_, size_, value.value<uint16_t>(),
                                 device_);
    break;

  case ScalarType::UNSIGNED_INT_32:
    darkside::fillData<uint32_t>(data_, size_, value.value<uint32_t>(),
                                 device_);
    break;

  case ScalarType::UNSIGNED_INT_64:
    darkside::fillData<uint64_t>(data_, size_, value.value<uint64_t>(),
                                 device_);
    break;

  default:
    break;
  }
}

void Storage::fillRandomData() {
  switch (scalar_type_) {
  case ScalarType::INT_8:
    darkside::fillRandomData<int8_t>(data_, size_, device_);
    break;
  case ScalarType::INT_16:
    darkside::fillRandomData<int16_t>(data_, size_, device_);
    break;
  case ScalarType::INT_32:
    darkside::fillRandomData<int32_t>(data_, size_, device_);
    break;
  case ScalarType::INT_64:
    darkside::fillRandomData<int64_t>(data_, size_, device_);
    break;
  case ScalarType::FLOAT_32:
    darkside::fillRandomData<float>(data_, size_, device_);
    break;
  case ScalarType::FLOAT_64:
    darkside::fillRandomData<double>(data_, size_, device_);
    break;
  case ScalarType::UNSIGNED_INT_8:
    darkside::fillRandomData<uint8_t>(data_, size_, device_);
    break;
  case ScalarType::UNSIGNED_INT_16:
    darkside::fillRandomData<uint16_t>(data_, size_, device_);
    break;
  case ScalarType::UNSIGNED_INT_32:
    darkside::fillRandomData<uint32_t>(data_, size_, device_);
    break;
  case ScalarType::UNSIGNED_INT_64:
    darkside::fillRandomData<uint64_t>(data_, size_, device_);
    break;
  default:
    break;
  }
}

void Storage::fillIncreaseData() {
  switch (scalar_type_) {
  case ScalarType::INT_8:
    darkside::fillIncreaseData<int8_t>(data_, size_, device_);
    break;
  case ScalarType::INT_16:
    darkside::fillIncreaseData<int16_t>(data_, size_, device_);
    break;
  case ScalarType::INT_32:
    darkside::fillIncreaseData<int32_t>(data_, size_, device_);
    break;
  case ScalarType::INT_64:
    darkside::fillIncreaseData<int64_t>(data_, size_, device_);
    break;
  case ScalarType::FLOAT_32:
    darkside::fillIncreaseData<float>(data_, size_, device_);
    break;
  case ScalarType::FLOAT_64:
    darkside::fillIncreaseData<double>(data_, size_, device_);
    break;
  case ScalarType::UNSIGNED_INT_8:
    darkside::fillIncreaseData<uint8_t>(data_, size_, device_);
    break;
  case ScalarType::UNSIGNED_INT_16:
    darkside::fillIncreaseData<uint16_t>(data_, size_, device_);
    break;
  case ScalarType::UNSIGNED_INT_32:
    darkside::fillIncreaseData<uint32_t>(data_, size_, device_);
    break;
  case ScalarType::UNSIGNED_INT_64:
    darkside::fillIncreaseData<uint64_t>(data_, size_, device_);
    break;
  default:
    break;
  }
}
void Storage::fillDecreaseData() {
  switch (scalar_type_) {
  case ScalarType::INT_8:
    darkside::fillDecreaseData<int8_t>(data_, size_, device_);
    break;
  case ScalarType::INT_16:
    darkside::fillDecreaseData<int16_t>(data_, size_, device_);
    break;
  case ScalarType::INT_32:
    darkside::fillDecreaseData<int32_t>(data_, size_, device_);
    break;
  case ScalarType::INT_64:
    darkside::fillDecreaseData<int64_t>(data_, size_, device_);
    break;
  case ScalarType::FLOAT_32:
    darkside::fillDecreaseData<float>(data_, size_, device_);
    break;
  case ScalarType::FLOAT_64:
    darkside::fillDecreaseData<double>(data_, size_, device_);
    break;
  case ScalarType::UNSIGNED_INT_8:
    darkside::fillDecreaseData<uint8_t>(data_, size_, device_);
    break;
  case ScalarType::UNSIGNED_INT_16:
    darkside::fillDecreaseData<uint16_t>(data_, size_, device_);
    break;
  case ScalarType::UNSIGNED_INT_32:
    darkside::fillDecreaseData<uint32_t>(data_, size_, device_);
    break;
  case ScalarType::UNSIGNED_INT_64:
    darkside::fillDecreaseData<uint64_t>(data_, size_, device_);
    break;
  default:
    break;
  }
};
} // namespace startorch
