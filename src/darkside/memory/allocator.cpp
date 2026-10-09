#include "darkside/memory/allocator.hpp"

#include <cstdint>
#include <new>

#include <cuda_runtime_api.h>
#include <driver_types.h>

#include "darkside/memory/buffer.hpp"
#include "startorch/common/types.hpp"

namespace darkside {

Allocator::Allocator(std::uint64_t bytes, startorch::MallocType buffer_type) {
  if (bytes == 0ul || buffer_type == startorch::MallocType::kUndefined)
    return;

  void *data = nullptr;

  switch (buffer_type) {
  case startorch::MallocType::kHost:
    data = static_cast<void *>(new (std::nothrow) std::uint8_t[bytes]);
    aligned_size_ = 64ul;
    break;

  case startorch::MallocType::kDevice:
    if (cudaMalloc(&data, bytes) != cudaSuccess)
      data = nullptr;
    aligned_size_ = 256ul;
    break;

  case startorch::MallocType::kPinned:
    if (cudaMallocHost(&data, bytes) != cudaSuccess)
      data = nullptr;
    aligned_size_ = 4096ul;
    break;

  case startorch::MallocType::kUnified:
    if (cudaMallocManaged(&data, bytes) != cudaSuccess)
      data = nullptr;
    aligned_size_ = 256ul;
    break;

  default:
    break;
  }

  if (data == nullptr) {
    aligned_size_ = 0ul;
    return;
  }

  buffer_ = Buffer(data, bytes, buffer_type);
}

Allocator::~Allocator() {
  if (buffer_.IsNull())
    return;

  void *data = buffer_.GetData();

  switch (buffer_.GetMallocType()) {
  case startorch::MallocType::kHost:
    delete[] static_cast<std::uint8_t *>(data);
    break;

  case startorch::MallocType::kPinned:
    cudaFreeHost(data);
    break;

  case startorch::MallocType::kDevice:
  case startorch::MallocType::kUnified:
    cudaFree(data);
    break;

  default:
    break;
  }
}

Allocator::operator bool() const {
  return !buffer_.IsNull() && aligned_size_ != 0ul;
}

Buffer &Allocator::GetBuffer() { return buffer_; }
const Buffer &Allocator::GetBuffer() const { return buffer_; }
std::uint64_t Allocator::GetOffset() const { return offset_; }
std::uint64_t Allocator::GetAlignedSize() const { return aligned_size_; }

startorch::MallocType Allocator::GetType() const {
  return buffer_.GetMallocType();
}

bool Allocator::IsNull() const { return !(*this); }

Buffer Allocator::NewBuffer(std::uint64_t bytes) {
  if (bytes == 0ul || buffer_.IsNull())
    return Buffer::GetNull();

  std::uint64_t aligned_bytes =
      (bytes + aligned_size_ - 1ul) & ~(aligned_size_ - 1ul);

  if (offset_ + aligned_bytes > buffer_.GetBytes())
    return Buffer::GetNull();

  std::uint8_t *data = static_cast<std::uint8_t *>(buffer_.GetData());
  void *new_data = static_cast<void *>(data + offset_);

  offset_ += aligned_bytes;

  return Buffer(new_data, bytes, buffer_.GetMallocType());
}

void Allocator::DeleteBuffer(Buffer &buffer) {
  if (buffer.IsNull() || buffer_.IsNull())
    return;

  const auto *data = static_cast<const std::uint8_t *>(buffer_.GetData());
  const auto *old_data = static_cast<const std::uint8_t *>(buffer.GetData());

  if (old_data < data || old_data >= data + buffer_.GetBytes())
    return;

  std::uint64_t current_offset = static_cast<std::uint64_t>(old_data - data);
  std::uint64_t aligned_bytes =
      (buffer.GetBytes() + aligned_size_ - 1ul) & ~(aligned_size_ - 1ul);

  if (current_offset + aligned_bytes == offset_)
    offset_ = current_offset;
}

Allocator &Allocator::GetNull() {
  static Allocator null_instance;
  return null_instance;
}

} // namespace darkside
