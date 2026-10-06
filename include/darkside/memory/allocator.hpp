#ifndef DARKSIDE_MEMORY_ALLOCATOR_HPP_
#define DARKSIDE_MEMORY_ALLOCATOR_HPP_

#include <cstdint>

#include "darkside/memory/buffer.hpp"
#include "startorch/common/types.hpp"

namespace darkside {

class Allocator {
public:
  Allocator() = default;
  Allocator(Allocator &&other) noexcept;
  Allocator(const Allocator &other) = delete;
  Allocator(std::uint64_t bytes, startorch::MallocType buffer_type);

  ~Allocator();

  Allocator &operator=(Allocator &&other) noexcept;
  Allocator &operator=(const Allocator &other) = delete;

  explicit operator bool() const;

  Buffer &GetBuffer();
  const Buffer &GetBuffer() const;
  std::uint64_t GetOffset() const;
  std::uint64_t GetAlignedSize() const;
  startorch::MallocType GetType() const;

  bool IsNull() const;

  Buffer NewBuffer(std::uint64_t bytes);
  void DeleteBuffer(Buffer &buffer);

private:
  Buffer buffer_ = Buffer();
  std::uint64_t offset_ = 0ul;
  std::uint64_t aligned_size_ = 0ul;
};

} // namespace darkside

#endif // !DARKSIDE_MEMORY_ALLOCATOR_HPP_
