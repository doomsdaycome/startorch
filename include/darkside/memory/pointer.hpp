#ifndef DARKSIDE_MEMORY_POINTER_HPP_
#define DARKSIDE_MEMORY_POINTER_HPP_

#include <memory>

#include "darkside/core/type.hpp"
#include "darkside/memory/value.hpp"

namespace darkside {

class Platform;

class Pointer {
 public:
  Pointer() = default;
  Pointer(const Pointer& other) = default;
  Pointer(Pointer&& other) noexcept = default;

  ~Pointer() = default;

  Pointer& operator=(const Pointer& other) = default;
  Pointer& operator=(Pointer&& other) noexcept = default;

 private:
  void* pointer_ = nullptr;
  ScalarType scalar_type_ = ScalarType::kUndefined;
  Value value_ = Value();
  std::shared_ptr<Platform> shared_platform_ = std::shared_ptr<Platform>();
};

}  // namespace darkside

#endif  // !DARKSIDE_MEMORY_POINTER_HPP_
