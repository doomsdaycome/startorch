#include "darkside/memory/allocator.hpp"

#include <algorithm>
#include <cstdint>
#include <new>
#include <utility>

#include <cuda_runtime_api.h>
#include <driver_types.h>

#include "darkside/memory/buffer.hpp"
#include "startorch/common/types.hpp"
#include "startorch/engine/memory.hpp"

namespace darkside {

Allocator::Allocator(Allocator &&other) noexcept
    : buffer_(std::move(other.buffer_)), offset_(other.offset_),
      aligned_size_(other.aligned_size_), memory_data_(other.memory_data_) {
  other.buffer_ = Buffer();
  other.offset_ = 0ul;
  other.aligned_size_ = 0ul;
  other.memory_data_ = nullptr;
}

Allocator::Allocator(std::uint64_t bytes, startorch::BufferType buffer_type,
                     startorch::Memory &memory) {
  if (!bytes || buffer_type == startorch::BufferType::kUndefined || !memory)
    return;

  void *data = nullptr;

  switch (buffer_type) {
  case startorch::BufferType::kHost:
    data = static_cast<void *>(new (std::nothrow) std::uint8_t[bytes]);
    aligned_size_ = 64ul;
    break;

  case startorch::BufferType::kDevice:
    if (cudaMalloc(&data, bytes) != cudaSuccess)
      data = nullptr;
    aligned_size_ = 256ul;
    break;

  case startorch::BufferType::kPinned:
    if (cudaMallocHost(&data, bytes) != cudaSuccess)
      data = nullptr;
    aligned_size_ = 4096ul;
    break;

  case startorch::BufferType::kUnified:
    if (cudaMallocManaged(&data, bytes) != cudaSuccess)
      data = nullptr;
    aligned_size_ = 256ul;
    break;

  default:
    break;
  }

  if (data == nullptr)
    return;

  buffer_ = Buffer(data, bytes, buffer_type, *this);
  memory_data_ = &memory;
}

Allocator::~Allocator() {
  if (!buffer_)
    return;

  void *data = buffer_.GetData();

  switch (buffer_.GetType()) {
  case startorch::BufferType::kHost:
    delete[] static_cast<std::uint8_t *>(data);
    break;

  case startorch::BufferType::kPinned:
    cudaFreeHost(data);
    break;

  case startorch::BufferType::kDevice:
  case startorch::BufferType::kUnified:
    cudaFree(data);
    break;

  default:
    break;
  }
}

Allocator &Allocator::operator=(Allocator &&other) noexcept {

  if (this != &other) {
    buffer_ = std::move(other.buffer_);
    offset_ = other.offset_;
    aligned_size_ = other.aligned_size_;
    memory_data_ = other.memory_data_;

    other.buffer_ = Buffer();
    other.offset_ = 0ul;
    other.aligned_size_ = 0ul;
    other.memory_data_ = nullptr;
  }

  return *this;
}

Allocator::operator bool() const { return buffer_ && *memory_data_; }
bool Allocator::operator!() const { return !(*this); }

Buffer &Allocator::GetBuffer() { return buffer_; }
const Buffer &Allocator::GetBuffer() const { return buffer_; }
std::uint64_t Allocator::GetOffset() const { return offset_; }
std::uint64_t Allocator::GetAlignedSize() const { return aligned_size_; }
startorch::Memory &Allocator::GetMemory() { return *memory_data_; }
const startorch::Memory &Allocator::GetMemory() const { return *memory_data_; }
bool Allocator::IsNull() const { return !(*this); }

Buffer Allocator::NewBuffer(std::uint64_t bytes) {
  if (bytes == 0ul || !buffer_)
    return Buffer();

  std::uint64_t aligned_offset =
      (offset_ + aligned_size_ - 1ul) & ~(aligned_size_ - 1ul);

  if (aligned_offset + bytes > buffer_.GetBytes())
    return Buffer();

  std::uint8_t *data = static_cast<std::uint8_t *>(buffer_.GetData());
  void *new_data = static_cast<void *>(data + aligned_offset);

  offset_ = aligned_offset + bytes;

  return Buffer(new_data, bytes, buffer_.GetType(), *this);
}

void Allocator::DeleteBuffer(const Buffer &buffer) {
  if (!buffer || !buffer_)
    return;

  const auto *data = static_cast<const std::uint8_t *>(buffer_.GetData());
  const auto *old_data = static_cast<const std::uint8_t *>(buffer.GetData());

  if (old_data < data || old_data >= data + buffer_.GetBytes())
    return;

  std::uint64_t aligned_offset = static_cast<std::uint64_t>(old_data - data);
  std::uint64_t bytes = buffer.GetBytes();

  if (aligned_offset + bytes == offset_) {
    offset_ = aligned_offset;

    while (!free_blocks_.empty() &&
           free_blocks_.back().start_offset + free_blocks_.back().bytes ==
               offset_) {
      offset_ = free_blocks_.back().start_offset;
      free_blocks_.pop_back();
    }

    if (free_blocks_.empty() && offset_ <= aligned_size_)
      offset_ = 0ul;
  } else {
    auto it = std::lower_bound(free_blocks_.begin(), free_blocks_.end(),
                               aligned_offset,
                               [](const FreeBlock &block, std::uint64_t value) {
                                 return block.start_offset < value;
                               });

    if (it != free_blocks_.end() && it->start_offset == aligned_offset)
      return;

    free_blocks_.insert(it, {aligned_offset, bytes});
  }
}

} // namespace darkside
