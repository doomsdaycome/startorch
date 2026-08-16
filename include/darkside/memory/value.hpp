#ifndef DARKSIDE_MEMORY_VALUE_HPP_
#define DARKSIDE_MEMORY_VALUE_HPP_

#include "darkside/core/types.hpp"

namespace darkside {

class Pointer;

class Value {
 public:
  Value() = default;
  Value(const Value& other) = default;
  Value(Value&& other) noexcept = default;

  ~Value() = default;

  Value& operator=(const Value& other) = default;
  Value& operator=(Value&& other) noexcept = default;

 private:
  ValueType value_ = false;
  Pointer* pointer_ = nullptr;
};

}  // namespace darkside

#endif  // !DARKSIDE_MEMORY_VALUE_HPP_
