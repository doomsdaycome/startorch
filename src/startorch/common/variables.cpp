#include "startorch/common/variables.hpp"

#include "startorch/common/literals.hpp"
#include "startorch/common/types.hpp"
#include "startorch/engine/system.hpp"

namespace startorch {

System kDefaultHostSystem(0, 3_GB, 1_GB, SystemType::kHost);
System kDefaultDeviceSystem(0, 768_MB, 256_MB, SystemType::kDevice);

}  // namespace startorch
