#ifndef STARTORCH_MEMORY_ALLOCATOR_HPP_
#define STARTORCH_MEMORY_ALLOCATOR_HPP_

#include <cstdint>

#include "startorch/common/types.hpp"
#include "startorch/memory/buffer.hpp"

namespace startorch {

class Allocator {
 public:
  Allocator();
  Allocator(Allocator&& other) noexcept = delete;
  Allocator(const Allocator& other) = delete;
  Allocator(std::uint64_t bytes, MallocType type);

  ~Allocator();

  Allocator& operator=(Allocator&& other) noexcept = delete;
  Allocator& operator=(const Allocator& other) = delete;

  explicit operator bool() const;

  Buffer& GetBuffer();
  const Buffer& GetBuffer() const;
  std::uint64_t GetOffset() const;
  std::uint64_t GetAlignedSize() const;
  MallocType GetType() const;

  bool IsNull() const;

  Buffer NewBuffer(std::uint64_t bytes);
  void DeleteBuffer(Buffer& buffer);

  static Allocator& GetNull();

 private:
  Buffer buffer_;
  std::uint64_t offset_;
  std::uint64_t aligned_size_;
};

}  // namespace startorch

#endif  // !STARTORCH_MEMORY_ALLOCATOR_HPP_
