#pragma once

#include "startorch/common.hpp"
#include "startorch/device.hpp"

#include <cstdint>
#include <type_traits>

namespace startorch {
class ScalarToCPP {
private:
  ScalarType scalar_type_ = ScalarType::UNSIGNED_INT_64;

  union {
    int64_t i_;
    double d_;
    uint64_t u_{0};
  };

public:
  ScalarToCPP() = default;
  template <typename T, typename = std::enable_if_t<std::is_integral_v<T>>>
  ScalarToCPP(T v) {
    if constexpr (std::is_signed_v<T>) {
      scalar_type_ = ScalarType::INT_64;
      i_ = static_cast<int64_t>(v);
    } else {
      scalar_type_ = ScalarType::UNSIGNED_INT_64;
      u_ = static_cast<uint64_t>(v);
    }
  }

  ScalarToCPP(double v) {
    scalar_type_ = ScalarType::FLOAT_64;
    d_ = v;
  }

  template <typename T> T value() const {
    switch (scalar_type_) {
    case ScalarType::INT_64:
      return static_cast<T>(i_);
    case ScalarType::UNSIGNED_INT_64:
      return static_cast<T>(u_);
    case ScalarType::FLOAT_64:
      return static_cast<T>(d_);
    default:
      return static_cast<T>(u_);
    }
  }
};

class Storage {
private:
  void *data_ = nullptr;
  uint64_t size_ = 0;
  ScalarType scalar_type_ = ScalarType::UNSIGNED_INT_8;
  Device device_ = Device();

public:
  Storage() = default;
  Storage(uint64_t size, ScalarType scalar_type, const Device &device);

  Storage(const Storage &other);
  Storage(Storage &&other) noexcept;

  ~Storage();

  Storage &operator=(const Storage &other);
  Storage &operator=(Storage &&other) noexcept;

  void *getData() const;
  uint64_t getSize() const;
  ScalarType getScalarType() const;
  const Device &getDevice() const;

  void setDevice(const Device &device);

  void fillData(const ScalarToCPP &value);
  void fillRandomData();
  void fillIncreaseData();
  void fillDecreaseData();
};
} // namespace startorch
