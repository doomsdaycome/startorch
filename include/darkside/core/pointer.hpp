#ifndef DARKSIDE_CORE_POINTER_HPP_
#define DARKSIDE_CORE_POINTER_HPP_

#include <cstdint>

#include "darkside/core/type.hpp"

namespace darkside {

class Value;
class Allocator;

class Pointer {
 public:
  Pointer() = default;
  Pointer(const Pointer& other) = default;
  Pointer(Pointer&& other) noexcept = default;

  ~Pointer() = default;

  Pointer(const Pointer& other, ScalarType scalar_type);
  Pointer(void* pointer, uint64_t size, ScalarType scalar_type,
          Allocator* allocator);

  Pointer& operator=(const Pointer& other) = default;
  Pointer& operator=(Pointer&& other) noexcept = default;

  bool operator!() const;
  bool operator<(const Pointer& other) const;
  bool operator>(const Pointer& other) const;
  bool operator==(const Pointer& other) const;
  bool operator!=(const Pointer& other) const;
  bool operator<=(const Pointer& other) const;
  bool operator>=(const Pointer& other) const;

  Pointer& operator++();
  Pointer operator++(int);
  Pointer& operator--();
  Pointer operator--(int);

  Pointer operator+(uint64_t offset);
  Pointer operator-(uint64_t offset);
  Pointer& operator+=(uint64_t offset);
  Pointer& operator-=(uint64_t offset);

  Value& operator*();
  const Value& operator*() const;

  Value& operator[](uint64_t index);
  const Value& operator[](uint64_t index) const;

  uint64_t GetSize() const;
  uint64_t GetScalarType() const;

 private:
  void* pointer_ = nullptr;
  uint64_t size_ = 0ul;
  ScalarType scalar_type_ = ScalarType::kUndefined;
  Value* value_ = nullptr;
  Allocator* allocator_ = nullptr;
};

Pointer operator+(uint64_t offset, const Pointer& pointer);
Pointer operator-(uint64_t offset, const Pointer& pointer);

}  // namespace darkside

#endif  // !DARKSIDE_CORE_POINTER_HPP_
