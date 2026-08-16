#ifndef DARKSIDE_CORE_PLATFORM_HPP_
#define DARKSIDE_CORE_PLATFORM_HPP_

namespace darkside {

class Platform {};

class HostPlatform : public Platform {};

class DevicePlatform : public Platform {};

}  // namespace darkside

#endif  // !DARKSIDE_CORE_PLATFORM_HPP_
