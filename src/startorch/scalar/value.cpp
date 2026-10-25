#include "startorch/scalar/value.hpp"

#include "startorch/memory/buffer.hpp"

namespace startorch {

Value::Value(Value&& other) noexcept
    : buffer_(other.buffer_),
      type_(other.type_),
      system_data_(other.system_data_) {
  other.buffer_ = Buffer();
}

// Value::Value(const Value& other);
// Value::Value(CPPType value);
// Value::Value(CPPType value, System& system);
// Value::Value(ScalarType scalar_type, System& system);
//
// Value::~Value();
//
// Value& Value::operator=(Value&&);
// Value& Value::operator=(const Value&);

}  // namespace startorch
