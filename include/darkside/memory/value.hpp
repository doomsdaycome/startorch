#ifndef DARKSIDE_MEMORY_VALUE_HPP_
#define DARKSIDE_MEMORY_VALUE_HPP_

#include "darkside/common/types.hpp"

namespace darkside {

class Pointer;

class Value {
 public:
  Value() = default;
  Value(const Value& other) = default;
  Value(Value&& other) noexcept = default;

  Value(ScalarValue scalar_value, ScalarType scalar_type, Pointer* pointer);

  ~Value() = default;

  Value& operator=(const Value& other) = default;
  Value& operator=(Value&& other) noexcept = default;

 private:
  ScalarValue scalar_value_ = false;
  ScalarType scalar_type_ = ScalarType::kUndefined;
  Pointer* pointer_ = nullptr;
};

}  // namespace darkside

#endif  // !DARKSIDE_MEMORY_VALUE_HPP_
