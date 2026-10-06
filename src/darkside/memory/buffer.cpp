#include "darkside/memory/buffer.hpp"

#include <cstdint>

#include "startorch/common/types.hpp"

namespace darkside {

Buffer::Buffer(void *data, std::uint64_t bytes, startorch::MallocType type)
    : data_(data), bytes_(bytes), type_(type) {
  if (data_ == nullptr || bytes_ == 0ul ||
      type_ == startorch::MallocType::kUndefined) {
    data_ = nullptr;
    bytes_ = 0ul;
    type_ = startorch::MallocType::kUndefined;
  }
}

Buffer::operator bool() const {
  return data_ != nullptr && bytes_ != 0ul &&
         type_ != startorch::MallocType::kUndefined &&
         type_ != startorch::MallocType::kOptionCount;
}

bool Buffer::operator!() const { return !static_cast<bool>(*this); }

void *Buffer::GetData() { return data_; }
const void *Buffer::GetData() const { return data_; }
std::uint64_t Buffer::GetBytes() const { return bytes_; }
startorch::MallocType Buffer::GetMallocType() const { return type_; }
bool Buffer::IsNull() const { return !(*this); }

} // namespace darkside
