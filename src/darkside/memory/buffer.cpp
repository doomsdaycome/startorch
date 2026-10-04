#include "darkside/memory/buffer.hpp"

#include <cstdint>

#include "startorch/common/types.hpp"

namespace darkside {

Buffer::Buffer(void *data, std::uint64_t bytes, startorch::BufferType type)
    : data_(data), bytes_(bytes), type_(type) {
  if (!data_ || !bytes_ || type_ == startorch::BufferType::kUndefined) {
    data_ = nullptr;
    bytes_ = 0ul;
    type_ = startorch::BufferType::kUndefined;
  }
}

Buffer::operator bool() const {
  return data_ && bytes_ && type_ != startorch::BufferType::kUndefined;
}

bool Buffer::operator!() const {
  return !data_ || !bytes_ || type_ == startorch::BufferType::kUndefined;
}

void *Buffer::GetData() { return data_; }
const void *Buffer::GetData() const { return data_; }
std::uint64_t Buffer::GetBytes() const { return bytes_; }
startorch::BufferType Buffer::GetType() const { return type_; }
bool Buffer::IsNull() const { return !(*this); }

} // namespace darkside
