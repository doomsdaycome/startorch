#ifndef STARTORCH_COMMON_VARIABLES_HPP_
#define STARTORCH_COMMON_VARIABLES_HPP_

#include "startorch/device/engine.hpp"
#include "startorch/device/system.hpp"

namespace startorch {

extern Engine kDefaultHostEngine;
extern Engine kDefaultDeviceEngine;

extern System kDefaultHostSystem;
extern System kDefaultDeviceSystem;

}  // namespace startorch

#endif  // !STARTORCH_COMMON_VARIABLES_HPP_
