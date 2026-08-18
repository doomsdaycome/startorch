#ifndef DARKSIDE_MEMORY_VALUE_HPP_
#define DARKSIDE_MEMORY_VALUE_HPP_

#include "darkside/common/types.hpp"
#include "darkside/macros/mapping.hpp"

namespace darkside {

class Pointer;

class Value {
 public:
  Value() = default;
  Value(Value&& other);
  Value(const Value& other);

#define DARKSIDE_INI_VALUE(C) Value(C value);
  DARKSIDE_FORALL_CPP_TYPE(DARKSIDE_INI_VALUE)
#undef DARKSIDE_INI_VALUE

  Value(const CPPType& value);
  Value(const CPPType& value, ScalarType scalar_type);
  Value(const CPPType& value, ScalarType scalar_type, Pointer* pointer);

  ~Value() = default;

  Value& operator=(Value&& other);
  Value& operator=(const Value& other);

  Pointer& operator&() &;
  const Pointer& operator&() const&;
  Pointer& operator&() && = delete;

  bool operator!() const;
  explicit operator bool() const;

  Value operator-() const;

  Value operator+(const Value& other) const;
  Value operator-(const Value& other) const;
  Value operator*(const Value& other) const;
  Value operator/(const Value& other) const;

  Value& operator+=(const Value& other);
  Value& operator-=(const Value& other);
  Value& operator*=(const Value& other);
  Value& operator/=(const Value& other);

  Value& operator++();
  Value& operator--();

  Value operator++(int);
  Value operator--(int);

  bool operator<(const Value& other) const;
  bool operator>(const Value& other) const;

  bool operator==(const Value& other) const;
  bool operator!=(const Value& other) const;
  bool operator<=(const Value& other) const;
  bool operator>=(const Value& other) const;

  template <typename T>
  T Get() const;

  ScalarType GetScalarType() const;

  Pointer* GetPointerRawPointer();
  const Pointer* GetPointerRawPointer() const;

 private:
  CPPType value_ = std::monostate{};
  ScalarType scalar_type_ = ScalarType::kUndefined;
  Pointer* pointer_raw_pointer_ = nullptr;
};

}  // namespace darkside

#endif  // !DARKSIDE_MEMORY_VALUE_HPP_
