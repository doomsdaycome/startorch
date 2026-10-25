#ifndef STARTORCH_SCALAR_VALUE_HPP_
#define STARTORCH_SCALAR_VALUE_HPP_

#include "startorch/common/types.hpp"
#include "startorch/engine/system.hpp"
#include "startorch/memory/buffer.hpp"

namespace startorch {

class Value {
 public:
  Value() = default;
  Value(Value&& other) noexcept;
  Value(const Value& other);
  Value(CPPType value);
  Value(CPPType value, System& system);
  Value(ScalarType scalar_type, System& system);

  ~Value();

  Value& operator=(Value&&);
  Value& operator=(const Value&);

 private:
  Buffer buffer_ = Buffer();
  ScalarType type_ = ScalarType::kUndefined;
  System* system_data_ = nullptr;
};

class ConstValue;

}  // namespace startorch

#endif  // !STARTORCH_SCALAR_VALUE_HPP_
