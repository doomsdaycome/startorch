#ifndef DARKSIDE_MEMORY_BUFFER_HPP_
#define DARKSIDE_MEMORY_BUFFER_HPP_

#include <cstdint>

#include "startorch/common/types.hpp"

namespace darkside {

class Buffer {
public:
  Buffer() = default;
  Buffer(Buffer &&other) noexcept = default;
  Buffer(const Buffer &other) = default;
  Buffer(void *data, std::uint64_t bytes, startorch::MallocType type);

  ~Buffer() = default;

  Buffer &operator=(Buffer &&other) noexcept = default;
  Buffer &operator=(const Buffer &other) = default;

  explicit operator bool() const;

  template <typename P> P *GetData();
  template <typename P> const P *GetData() const;

  void *GetData();
  const void *GetData() const;
  std::uint64_t GetBytes() const;
  startorch::MallocType GetMallocType() const;

  bool IsNull() const;

private:
  void *data_ = nullptr;
  std::uint64_t bytes_ = 0ul;
  startorch::MallocType type_ = startorch::MallocType::kUndefined;
};

} // namespace darkside

#endif // !DARKSIDE_MEMORY_BUFFER_HPP_
