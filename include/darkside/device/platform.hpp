#ifndef DARKSIDE_PLATFORM_PLATFORM_HPP_
#define DARKSIDE_PLATFORM_PLATFORM_HPP_

#include <memory>
namespace darkside {

class Platform {};

class HostPlatform : public Platform {};

class DevicePlatform : public Platform {};

class PlatformPair {
 public:
 private:
};

}  // namespace darkside

#endif  // !DARKSIDE_CORE_PLATFORM_HPP_
