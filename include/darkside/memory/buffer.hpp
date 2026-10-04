#ifndef DARKSIDE_MEMORY_BUFFER_HPP_
#define DARKSIDE_MEMORY_BUFFER_HPP_

#include <cstdint>

#include "startorch/common/types.hpp"

namespace darkside {

class Buffer {
public:
  Buffer() = default;
  Buffer(Buffer &&other);
  Buffer(const Buffer &other) = delete;
  Buffer(void *data, std::uint64_t bytes, startorch::BufferType type);

  ~Buffer() = default;

  Buffer &operator=(Buffer &&other);
  Buffer &operator=(const Buffer &other) = delete;

  void *GetData();
  const void *GetData() const;
  std::uint64_t GetBytes() const;
  startorch::BufferType GetType() const;

private:
  void *data_ = nullptr;
  std::uint64_t bytes_ = 0ul;
  startorch::BufferType type_ = startorch::BufferType::kUndefined;
};

} // namespace darkside

#endif // !DARKSIDE_MEMORY_BUFFER_HPP_
