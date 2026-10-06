#include "startorch/common/defaults.hpp"

#include "startorch/common/literals.hpp"
#include "startorch/common/types.hpp"
#include "startorch/engine/system.hpp"

namespace startorch {

System &GetDefaultHostSystem() {
  static System default_host_system(0, 1536_MB, 512_MB, SystemType::kHost);
  return default_host_system;
}

System &GetDefaultDeviceSystem() {
  static System default_device_system(0, 768_MB, 256_MB, SystemType::kDevice);
  return default_device_system;
}

} // namespace startorch
