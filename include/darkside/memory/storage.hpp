#ifndef DARKSIDE_MEMORY_ARENA_HPP_
#define DARKSIDE_MEMORY_ARENA_HPP_

#include <cstdint>

#include "darkside/memory/pointer.hpp"

namespace darkside {

class Platform;

class Storage {
 public:
  Storage() = default;
  Storage(const Storage& other) = default;
  Storage(Storage&& other) noexcept = default;

  ~Storage() = default;

  Storage& operator=(const Storage& other) = default;
  Storage& operator=(Storage&& other) noexcept = default;

 private:
  Pointer pointer_ = Pointer();
  uint64_t size_ = 0ul;
};

}  // namespace darkside

#endif  // !DARKSIDE_MEMORY_ARENA_HPP_
