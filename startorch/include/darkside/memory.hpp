#pragma once

#include "startorch/device.hpp"

#include <cstdint>

namespace darkside {
void *makeData(uint64_t size, const startorch::Device &device);
void freeData(void *pointer, const startorch::Device &device);
void copyData(void *destination, void *source, uint64_t size,
              const startorch::DevicePair &device_pair);
} // namespace startorch
