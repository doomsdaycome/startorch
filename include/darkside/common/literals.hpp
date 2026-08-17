#ifndef DARKSIDE_COMMON_LITERALS_HPP_
#define DARKSIDE_COMMON_LITERALS_HPP_

#include <cstdint>

namespace darkside {

constexpr std::uint64_t operator""_KiB(unsigned long long value) noexcept {
  return value << 10;
}

constexpr std::uint64_t operator""_MiB(unsigned long long value) noexcept {
  return value << 20;
}

constexpr std::uint64_t operator""_GiB(unsigned long long value) noexcept {
  return value << 30;
}

}  // namespace darkside

#endif  // !DARKSIDE_COMMON_LITERALS_HPP_
