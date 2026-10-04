#include "darkside/memory/buffer.hpp"

#include <cstdint>

#include "startorch/common/types.hpp"

namespace darkside {

Buffer::Buffer(Buffer &&other) noexcept
    : data_(other.data_), bytes_(other.bytes_), type_(other.type_) {
  other.data_ = nullptr;
  other.bytes_ = 0ul;
  other.type_ = startorch::BufferType::kUndefined;
}

Buffer::Buffer(void *data, std::uint64_t bytes, startorch::BufferType type)
    : data_(data), bytes_(bytes), type_(type) {
  if (data_ == nullptr || bytes_ == 0ul ||
      type_ == startorch::BufferType::kUndefined) {
    data_ = nullptr;
    bytes_ = 0ul;
    type_ = startorch::BufferType::kUndefined;
  }
}

Buffer &Buffer::operator=(Buffer &&other) noexcept {
  if (this != &other) {
    data_ = other.data_;
    bytes_ = other.bytes_;
    type_ = other.type_;

    other.data_ = nullptr;
    other.bytes_ = 0ul;
    other.type_ = startorch::BufferType::kUndefined;
  }

  return *this;
}

void *Buffer::GetData() { return data_; }
const void *Buffer::GetData() const { return data_; }
std::uint64_t Buffer::GetBytes() const { return bytes_; }
startorch::BufferType Buffer::GetType() const { return type_; }

} // namespace darkside
