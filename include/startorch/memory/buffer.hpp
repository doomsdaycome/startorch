#ifndef STARTORCH_MEMORY_BUFFER_HPP_
#define STARTORCH_MEMORY_BUFFER_HPP_

#include <cstdint>

#include "startorch/common/types.hpp"

namespace startorch {

class Buffer {
 public:
  Buffer();
  Buffer(Buffer&& other) noexcept = default;
  Buffer(const Buffer& other) = default;
  Buffer(void* data, std::uint64_t bytes, MallocType malloc_type);

  ~Buffer() = default;

  Buffer& operator=(Buffer&& other) noexcept = default;
  Buffer& operator=(const Buffer& other) = default;

  explicit operator bool() const;

  template <typename P>
  P* GetData();
  template <typename P>
  const P* GetData() const;

  void* GetData();
  const void* GetData() const;
  std::uint64_t GetBytes() const;
  MallocType GetMallocType() const;

  bool IsNull() const;

  static Buffer& GetNull();

 private:
  void* data_;
  std::uint64_t bytes_;
  MallocType malloc_type_;
};

}  // namespace startorch

#endif  // !STARTORCH_MEMORY_BUFFER_HPP_
