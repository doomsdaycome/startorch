#include <cstdint>
#include <iostream>

#include "startorch/common/literals.hpp"
#include "startorch/common/types.hpp"
#include "startorch/common/utilities.hpp"
#include "startorch/engine/system.hpp"

int main(int argc, char* argv[]) {
  std::cout << "Sigillum Dei Aemeath" << std::endl;

  startorch::System cpu(0, 512_KB, 128_KB, startorch::SystemType::kHost);
  startorch::System gpu(0, 512_KB, 128_KB, startorch::SystemType::kDevice);

  auto buffer = cpu.GetMemory().GetPhysicalAllocator().NewBuffer(
      startorch::SizeOfScalarType(startorch::ScalarType::kUnsignedInt32));

  buffer.GetData<uint32_t>();

  return 0;
}
