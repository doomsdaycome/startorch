#ifndef DARKSIDE_MEMORY_VALUE_HPP_
#define DARKSIDE_MEMORY_VALUE_HPP_

#include <memory>

#include "darkside/core/type.hpp"

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
  std::weak_ptr<Pointer> weak_pointer_ = std::weak_ptr<Pointer>();
};

}  // namespace darkside

#endif  // !DARKSIDE_MEMORY_VALUE_HPP_
