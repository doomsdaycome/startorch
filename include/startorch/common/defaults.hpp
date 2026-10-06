#ifndef STARTORCH_COMMON_DEFAULTS_HPP_
#define STARTORCH_COMMON_DEFAULTS_HPP_

#include "startorch/engine/system.hpp"

namespace startorch {

System &GetDefaultHostSystem();
System &GetDefaultDeviceSystem();

} // namespace startorch

#endif // !STARTORCH_COMMON_DEFAULTS_HPP_
