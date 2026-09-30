#ifndef STARTORCH_COMMON_LITERALS_HPP_
#define STARTORCH_COMMON_LITERALS_HPP_

#include <cstdint>

namespace startorch {

constexpr std::uint64_t operator""_KB(unsigned long long size) noexcept {
  return static_cast<std::uint64_t>(size) << 10;
}

constexpr std::uint64_t operator""_MB(unsigned long long size) noexcept {
  return static_cast<std::uint64_t>(size) << 20;
}

constexpr std::uint64_t operator""_GB(unsigned long long size) noexcept {
  return static_cast<std::uint64_t>(size) << 30;
}

} // namespace startorch

#endif // !STARTORCH_COMMON_LITERALS_HPP_
