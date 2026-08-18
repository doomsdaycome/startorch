#ifndef DARKSIDE_MEMORY_POINTER_HPP_
#define DARKSIDE_MEMORY_POINTER_HPP_

#include <cstdint>

#include "darkside/common/types.hpp"
#include "darkside/memory/value.hpp"

namespace darkside {

class Arena;

class Pointer {
 public:
  Pointer() = default;
  Pointer(Pointer&& other) = default;
  Pointer(const Pointer& other) = default;

#define DARKSIDE_INI_POINTER(C) Pointer(C* raw_pointer);
  DARKSIDE_FORALL_CPP_TYPE(DARKSIDE_INI_POINTER)
#undef DARKSIDE_INI_POINTER

  Pointer(void* raw_pointer);
  Pointer(void* raw_pointer, ScalarType scalar_type);
  Pointer(void* raw_pointer, ScalarType scalar_type, Arena* arena_pointer);

  ~Pointer() = default;

  Pointer& operator=(Pointer&& other) = default;
  Pointer& operator=(const Pointer& other) = default;

  Value& operator*() &;
  const Value& operator*() const&;
  Value& operator*() && = delete;

  Value* operator->();
  const Value* operator->() const;

  bool operator!() const;
  explicit operator bool() const;

  Pointer operator+(uint64_t offset) const;
  Pointer operator-(uint64_t offset) const;

  uint64_t operator-(const Pointer& other) const;

  Pointer& operator+=(uint64_t offset);
  Pointer& operator-=(uint64_t offset);

  Pointer& operator++();
  Pointer& operator--();

  Pointer operator++(int);
  Pointer operator--(int);

  bool operator<(const Pointer& other) const;
  bool operator>(const Pointer& other) const;

  bool operator==(const Pointer& other) const;
  bool operator!=(const Pointer& other) const;
  bool operator<=(const Pointer& other) const;
  bool operator>=(const Pointer& other) const;

  void* GetRawPointer();
  const void* GetRawPointer() const;

  ScalarType GetScalarType() const;

  Arena* GetArenaRawPointer();
  const Arena* GetArenaRawPointer() const;

 private:
  void* raw_pointer_ = nullptr;
  ScalarType scalar_type_ = ScalarType::kUndefined;
  Value value_ = Value();
  Arena* arena_raw_pointer_ = nullptr;
};

}  // namespace darkside

#endif  // !DARKSIDE_MEMORY_POINTER_HPP_
