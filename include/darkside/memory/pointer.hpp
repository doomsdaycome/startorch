#ifndef DARKSIDE_MEMORY_POINTER_HPP_
#define DARKSIDE_MEMORY_POINTER_HPP_

#include "darkside/core/types.hpp"
#include "darkside/memory/value.hpp"

namespace darkside {

class Arena;

class Pointer {
 public:
  Pointer() = default;
  Pointer(const Pointer& other) = default;
  Pointer(Pointer&& other) noexcept = default;

  Pointer(void* pointer, ScalarType scalar_type, Arena* arena_pointer);

  ~Pointer() = default;

  Pointer& operator=(const Pointer& other) = default;
  Pointer& operator=(Pointer&& other) noexcept = default;

 private:
  void* pointer_ = nullptr;
  ScalarType scalar_type_ = ScalarType::kUndefined;
  Value value_ = Value();
  Arena* arena_pointer_ = nullptr;
};

}  // namespace darkside

#endif  // !DARKSIDE_MEMORY_POINTER_HPP_
