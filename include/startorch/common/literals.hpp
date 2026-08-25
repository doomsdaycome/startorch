#ifndef STARTORCH_COMMON_LITERALS_HPP_
#define STARTORCH_COMMON_LITERALS_HPP_

#include <cstdint>

constexpr std::uint64_t operator""_KiB(unsigned long long value) noexcept {
  return value << 10;
}

constexpr std::uint64_t operator""_MiB(unsigned long long value) noexcept {
  return value << 20;
}

constexpr std::uint64_t operator""_GiB(unsigned long long value) noexcept {
  return value << 30;
}

#endif  // !STARTORCH_COMMON_LITERALS_HPP_
