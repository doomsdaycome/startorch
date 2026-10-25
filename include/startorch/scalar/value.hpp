#ifndef STARTORCH_SCALAR_VALUE_HPP_
#define STARTORCH_SCALAR_VALUE_HPP_

#include "startorch/common/types.hpp"
#include "startorch/engine/system.hpp"
#include "startorch/memory/buffer.hpp"

namespace startorch {

class Value {
 public:
  Value();
  Value(Value&& other) noexcept;
  Value(const Value& other);
  Value(ScalarType type, System& system);

  ~Value();

  Value& operator=(Value&& other) noexcept;
  Value& operator=(const Value& other);

  explicit operator bool() const;

  bool IsNull() const;

  static Value& GetNull();

 private:
  Buffer buffer_;
  ScalarType type_;
  System* system_data_;
};

class ConstValue;

}  // namespace startorch

#endif  // !STARTORCH_SCALAR_VALUE_HPP_
