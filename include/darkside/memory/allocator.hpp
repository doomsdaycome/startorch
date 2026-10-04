#ifndef DARKSIDE_MEMORY_ALLOCATOR_HPP_
#define DARKSIDE_MEMORY_ALLOCATOR_HPP_

#include <cstdint>
#include <vector>

#include "darkside/memory/buffer.hpp"
#include "startorch/common/types.hpp"

namespace startorch {

class Memory;

} // namespace startorch

namespace darkside {

struct FreeBlock {
  std::uint64_t start_offset;
  std::uint64_t bytes;
};

class Allocator {
public:
  Allocator() = default;
  Allocator(Allocator &&other) noexcept;
  Allocator(const Allocator &other) = delete;
  Allocator(std::uint64_t bytes, startorch::BufferType buffer_type,
            startorch::Memory &memory);

  ~Allocator();

  Allocator &operator=(Allocator &&other) noexcept;
  Allocator &operator=(const Allocator &other) = delete;

  explicit operator bool() const;
  bool operator!() const;

  Buffer &GetBuffer();
  const Buffer &GetBuffer() const;
  std::uint64_t GetOffset() const;
  std::uint64_t GetAlignedSize() const;
  startorch::Memory &GetMemory();
  const startorch::Memory &GetMemory() const;

  bool IsNull() const;

  Buffer NewBuffer(std::uint64_t bytes);
  void DeleteBuffer(const Buffer &buffer);

private:
  Buffer buffer_ = Buffer();
  std::uint64_t offset_ = 0ul;
  std::uint64_t aligned_size_ = 0ul;
  startorch::Memory *memory_data_ = nullptr;
  std::vector<FreeBlock> free_blocks_ = std::vector<FreeBlock>();
};

} // namespace darkside

#endif // !DARKSIDE_MEMORY_ALLOCATOR_HPP_
