#ifndef DARKSIDE_MEMORY_BUFFER_HPP_
#define DARKSIDE_MEMORY_BUFFER_HPP_

#include <cstdint>

#include "startorch/common/types.hpp"

namespace darkside {

class Allocator;

class Buffer {
public:
  Buffer() = default;
  Buffer(Buffer &&other) noexcept;
  Buffer(const Buffer &other) = delete;
  Buffer(void *data, std::uint64_t bytes, startorch::BufferType type,
         Allocator &allocator);

  ~Buffer() = default;

  Buffer &operator=(Buffer &&other) noexcept;
  Buffer &operator=(const Buffer &other) = delete;

  explicit operator bool() const;
  bool operator!() const;

  void *GetData();
  const void *GetData() const;
  std::uint64_t GetBytes() const;
  startorch::BufferType GetType() const;
  Allocator &GetAllocator();
  const Allocator &GetAllocator() const;

  bool IsNull() const;

private:
  void *data_ = nullptr;
  std::uint64_t bytes_ = 0ul;
  startorch::BufferType type_ = startorch::BufferType::kUndefined;
  Allocator *allocator_data_ = nullptr;
};

} // namespace darkside

#endif // !DARKSIDE_MEMORY_BUFFER_HPP_
