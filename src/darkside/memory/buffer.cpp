#include "darkside/memory/buffer.hpp"

#include <cstdint>

#include "darkside/memory/allocator.hpp"
#include "startorch/common/types.hpp"

namespace darkside {

Buffer::Buffer(Buffer &&other) noexcept
    : data_(other.data_), bytes_(other.bytes_), type_(other.type_),
      allocator_data_(other.allocator_data_) {
  other.data_ = nullptr;
  other.bytes_ = 0ul;
  other.type_ = startorch::BufferType::kUndefined;
  other.allocator_data_ = nullptr;
}

Buffer::Buffer(void *data, std::uint64_t bytes, startorch::BufferType type,
               Allocator &allocator)
    : data_(data), bytes_(bytes), type_(type), allocator_data_(&allocator) {
  if (!data_ || !bytes_ || type_ == startorch::BufferType::kUndefined ||
      !(*allocator_data_)) {
    data_ = nullptr;
    bytes_ = 0ul;
    type_ = startorch::BufferType::kUndefined;
    allocator_data_ = nullptr;
  }
}

Buffer &Buffer::operator=(Buffer &&other) noexcept {
  if (this != &other) {
    data_ = other.data_;
    bytes_ = other.bytes_;
    type_ = other.type_;
    allocator_data_ = other.allocator_data_;

    other.data_ = nullptr;
    other.bytes_ = 0ul;
    other.type_ = startorch::BufferType::kUndefined;
    other.allocator_data_ = nullptr;
  }

  return *this;
}

Buffer::operator bool() const {
  return data_ && bytes_ && type_ != startorch::BufferType::kUndefined &&
         *allocator_data_;
}

bool Buffer::operator!() const { return !(*this); }

void *Buffer::GetData() { return data_; }
const void *Buffer::GetData() const { return data_; }
std::uint64_t Buffer::GetBytes() const { return bytes_; }
startorch::BufferType Buffer::GetType() const { return type_; }
Allocator &Buffer::GetAllocator() { return *allocator_data_; }
const Allocator &Buffer::GetAllocator() const { return *allocator_data_; }
bool Buffer::IsNull() const { return !(*this); }

} // namespace darkside
