#ifndef DARKSIDE_MEMORY_MEMORY_HPP_
#define DARKSIDE_MEMORY_MEMORY_HPP_

#include "darkside/memory/allocator.hpp"

namespace darkside {

class Memory {
 public:
 private:
};

class HostMemory : public Memory {};

class DeviceMemory : public Memory {};

}  // namespace darkside

#endif  // !DARKSIDE_MEMORY_MEMORY_HPP_
