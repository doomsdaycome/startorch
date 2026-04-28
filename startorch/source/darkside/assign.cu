#include "startorch/common.hpp"
#include "startorch/device.hpp"
#include "startorch/random.hpp"

#include "darkside/assign.hpp"
#include "darkside/common.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>

#include <ctime>
#include <curand_kernel.h>

namespace darkside {
template <typename T>
__global__ void fillDataGPU(T *data, uint64_t size, T value) {
  uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size)
    data[idx] = value;
}

template <typename T> void fillDataCPU(T *data, uint64_t size, T value) {
  if (size == 0 || data == nullptr)
    return;
  if (value == (T)0) {
    std::memset(data, 0, size * sizeof(T));
    return;
  }
  std::fill_n(data, size, value);
}

template <typename T>
__global__ void fillRandomDataGPU(T *data, uint64_t size, uint64_t seed) {
  uint64_t id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id < size) {
    curandState state;
    curand_init(seed, id, 0, &state);

    if constexpr (std::is_same_v<T, float>)
      data[id] = curand_uniform(&state);
    else if constexpr (std::is_same_v<T, double>)
      data[id] = curand_uniform_double(&state);
    else
      data[id] = static_cast<T>(curand(&state));
  }
}

template <typename T> void fillRandomDataCPU(T *data, uint64_t size) {
  for (uint64_t i = 0; i < size; i++)
    data[i] = pcg32_convert<T>::convert();
}

template <typename T>
__global__ void fillIncreaseDataGPU(T *data, uint64_t size) {
  uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size)
    data[idx] = (T)idx;
}

template <typename T> void fillIncreaseDataCPU(T *data, uint64_t size) {
  if (size == 0 || data == nullptr)
    return;
  for (uint64_t i = 0; i < size; i++)
    data[i] = (T)i;
}

template <typename T>
__global__ void fillDecreaseDataGPU(T *data, uint64_t size) {
  uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size)
    data[idx] = (T)(size - 1 - idx);
}

template <typename T> void fillDecreaseDataCPU(T *data, uint64_t size) {
  if (size == 0 || data == nullptr)
    return;
  for (uint64_t i = 0; i < size; i++)
    data[i] = (T)(size - 1 - i);
}

template <typename T>
void fillData(void *data, uint64_t size, T value,
              const startorch::Device &device) {
  switch (device.getDeviceType()) {
  case startorch::DeviceType::CPU:
    fillDataCPU<T>((T *)data, size, value);
    break;

  case startorch::DeviceType::GPU:
    fillDataGPU<T><<<BLOCKS(size), THREADS>>>((T *)data, size, value);
    break;

  default:
    break;
  }
}

template void fillData<int8_t>(void *, uint64_t, int8_t,
                               const startorch::Device &);
template void fillData<int16_t>(void *, uint64_t, int16_t,
                                const startorch::Device &);
template void fillData<int32_t>(void *, uint64_t, int32_t,
                                const startorch::Device &);
template void fillData<int64_t>(void *, uint64_t, int64_t,
                                const startorch::Device &);
template void fillData<float>(void *, uint64_t, float,
                              const startorch::Device &);
template void fillData<double>(void *, uint64_t, double,
                               const startorch::Device &);
template void fillData<uint8_t>(void *, uint64_t, uint8_t,
                                const startorch::Device &);
template void fillData<uint16_t>(void *, uint64_t, uint16_t,
                                 const startorch::Device &);
template void fillData<uint32_t>(void *, uint64_t, uint32_t,
                                 const startorch::Device &);
template void fillData<uint64_t>(void *, uint64_t, uint64_t,
                                 const startorch::Device &);



template <typename T>
void fillRandomData(void *data, uint64_t size,
                    const startorch::Device &device) {
  switch (device.getDeviceType()) {
  case startorch::DeviceType::CPU:
    fillRandomDataCPU<T>((T *)data, size);
    break;

  case startorch::DeviceType::GPU:
    fillRandomDataGPU<T><<<BLOCKS(size), THREADS>>>((T *)data, size, time(nullptr));
    break;
  }
}

template void fillRandomData<int8_t>(void *, uint64_t,
                                       const startorch::Device &);
template void fillRandomData<int16_t>(void *, uint64_t,
                                        const startorch::Device &);
template void fillRandomData<int32_t>(void *, uint64_t,
                                        const startorch::Device &);
template void fillRandomData<int64_t>(void *, uint64_t,
                                        const startorch::Device &);
template void fillRandomData<float>(void *, uint64_t,
                                      const startorch::Device &);
template void fillRandomData<double>(void *, uint64_t,
                                       const startorch::Device &);
template void fillRandomData<uint8_t>(void *, uint64_t,
                                        const startorch::Device &);
template void fillRandomData<uint16_t>(void *, uint64_t,
                                         const startorch::Device &);
template void fillRandomData<uint32_t>(void *, uint64_t,
                                         const startorch::Device &);
template void fillRandomData<uint64_t>(void *, uint64_t,
                                         const startorch::Device &);

template <typename T>
void fillIncreaseData(void *data, uint64_t size,
                      const startorch::Device &device) {
  switch (device.getDeviceType()) {
  case startorch::DeviceType::CPU:
    fillIncreaseDataCPU<T>((T *)data, size);
    break;

  case startorch::DeviceType::GPU:
    fillIncreaseDataGPU<T><<<BLOCKS(size), THREADS>>>((T *)data, size);
    break;

  default:
    break;
  }
}

template void fillIncreaseData<int8_t>(void *, uint64_t,
                                       const startorch::Device &);
template void fillIncreaseData<int16_t>(void *, uint64_t,
                                        const startorch::Device &);
template void fillIncreaseData<int32_t>(void *, uint64_t,
                                        const startorch::Device &);
template void fillIncreaseData<int64_t>(void *, uint64_t,
                                        const startorch::Device &);
template void fillIncreaseData<float>(void *, uint64_t,
                                      const startorch::Device &);
template void fillIncreaseData<double>(void *, uint64_t,
                                       const startorch::Device &);
template void fillIncreaseData<uint8_t>(void *, uint64_t,
                                        const startorch::Device &);
template void fillIncreaseData<uint16_t>(void *, uint64_t,
                                         const startorch::Device &);
template void fillIncreaseData<uint32_t>(void *, uint64_t,
                                         const startorch::Device &);
template void fillIncreaseData<uint64_t>(void *, uint64_t,
                                         const startorch::Device &);

template <typename T>
void fillDecreaseData(void *data, uint64_t size,
                      const startorch::Device &device) {
  switch (device.getDeviceType()) {
  case startorch::DeviceType::CPU:
    fillDecreaseDataCPU<T>((T *)data, size);
    break;

  case startorch::DeviceType::GPU:
    fillDecreaseDataGPU<T><<<BLOCKS(size), THREADS>>>((T *)data, size);
    break;

  default:
    break;
  }
}

template void fillDecreaseData<int8_t>(void *, uint64_t,
                                       const startorch::Device &);
template void fillDecreaseData<int16_t>(void *, uint64_t,
                                        const startorch::Device &);
template void fillDecreaseData<int32_t>(void *, uint64_t,
                                        const startorch::Device &);
template void fillDecreaseData<int64_t>(void *, uint64_t,
                                        const startorch::Device &);
template void fillDecreaseData<float>(void *, uint64_t,
                                      const startorch::Device &);
template void fillDecreaseData<double>(void *, uint64_t,
                                       const startorch::Device &);
template void fillDecreaseData<uint8_t>(void *, uint64_t,
                                        const startorch::Device &);
template void fillDecreaseData<uint16_t>(void *, uint64_t,
                                         const startorch::Device &);
template void fillDecreaseData<uint32_t>(void *, uint64_t,
                                         const startorch::Device &);
template void fillDecreaseData<uint64_t>(void *, uint64_t,
                                         const startorch::Device &);
} // namespace darkside
