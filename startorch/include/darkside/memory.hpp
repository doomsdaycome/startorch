#pragma once

#include "startorch/common.hpp"
#include "startorch/device.hpp"

#include <cstdint>

namespace darkside {
void *makeMemory(uint64_t size, const startorch::Device &device);
void freeMemory(void *pointer, const startorch::Device &device);
void copyMemory(void *destination, void *source,
                const startorch::DevicePair &device_pair);

class Storage {
private:
  void *data_ = nullptr;
  uint64_t size_ = 0;
  startorch::ScalarType scalar_type_ = startorch::ScalarType::UNSIGNED_INT_8;
  startorch::Device device_ = startorch::Device();

public:
  Storage() = default;
  Storage(uint64_t size, startorch::ScalarType scalar_type,
          const startorch::Device &device);

  Storage(const Storage &other);
  Storage(Storage &&other) noexcept;

  ~Storage();

  Storage &operator=(const Storage &other);
  Storage &operator=(Storage &&other) noexcept;

  void *getData() const;
  uint64_t getSize() const;
  startorch::ScalarType getScalarType() const;
  const startorch::Device &getDevice() const;
};

uint64_t getScalarSize(startorch::ScalarType scalar_type);
uint64_t getStorageSize(const Storage &buffer);
} // namespace darkside
