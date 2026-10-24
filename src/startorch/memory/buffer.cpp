#include "startorch/memory/buffer.hpp"

#include <cstdint>

#include "startorch/common/types.hpp"
#include "startorch/macros/mapping.hpp"

#define STARTORCH_DEF_BUFFER_GET_DATA(C) \
  template C* Buffer::GetData<C>();      \
  template const C* Buffer::GetData<C>() const;

namespace startorch {

Buffer::Buffer(void* data, std::uint64_t bytes, MallocType malloc_type)
    : data_(data), bytes_(bytes), malloc_type_(malloc_type) {
  if (data_ == nullptr || bytes_ == 0ul ||
      malloc_type_ == MallocType::kUndefined ||
      malloc_type_ == MallocType::kOptionCount) {
    data_ = nullptr;
    bytes_ = 0ul;
    malloc_type_ = MallocType::kUndefined;
  }
}

Buffer::operator bool() const {
  return data_ != nullptr && bytes_ != 0ul &&
         malloc_type_ != MallocType::kUndefined &&
         malloc_type_ != MallocType::kOptionCount;
}

template <typename P>
P* Buffer::GetData() {
  return static_cast<P*>(data_);
}

template <typename P>
const P* Buffer::GetData() const {
  return static_cast<const P*>(data_);
}

STARTORCH_FORALL_CPP_TYPE(STARTORCH_DEF_BUFFER_GET_DATA)

void* Buffer::GetData() { return data_; }

const void* Buffer::GetData() const { return data_; }

std::uint64_t Buffer::GetBytes() const { return bytes_; }

MallocType Buffer::GetMallocType() const { return malloc_type_; }

bool Buffer::IsNull() const { return !(*this); }

Buffer& Buffer::GetNull() {
  static Buffer null_instance;
  return null_instance;
}

}  // namespace startorch
