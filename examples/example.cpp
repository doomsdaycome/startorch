#include <cstdint>
#include <iostream>

#include "startorch/common/types.hpp"
#include "startorch/common/utilities.hpp"
#include "startorch/common/variables.hpp"
#include "startorch/engine/bridge.hpp"
#include "startorch/engine/system.hpp"

int main(int argc, char* argv[]) {
  uint8_t sym[] = "Sigillum Dei Aemeath";
  uint64_t len = 20ul;

  auto src = startorch::kDefaultHostSystem.GetMemory()
                 .GetPhysicalAllocator()
                 .NewBuffer(startorch::SizeOfScalarType(
                                startorch::ScalarType::kUnsignedInt8) *
                            len);

  auto dst = startorch::kDefaultDeviceSystem.GetMemory()
                 .GetPhysicalAllocator()
                 .NewBuffer(startorch::SizeOfScalarType(
                                startorch::ScalarType::kUnsignedInt8) *
                            len);

  auto res = startorch::kDefaultHostSystem.GetMemory()
                 .GetPhysicalAllocator()
                 .NewBuffer(startorch::SizeOfScalarType(
                                startorch::ScalarType::kUnsignedInt8) *
                            len);

  for (uint64_t i = 0; i < len; i++) src.GetData<uint8_t>()[i] = sym[i];

  startorch::Bridge(startorch::kDefaultDeviceSystem,
                    startorch::kDefaultHostSystem)
      .Copy(dst, src);

  startorch::Bridge(startorch::kDefaultHostSystem,
                    startorch::kDefaultDeviceSystem)
      .Copy(res, dst);

  std::cout << res.GetData<uint8_t>();

  return 0;
}
