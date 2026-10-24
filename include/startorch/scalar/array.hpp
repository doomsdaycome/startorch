#ifndef STARTORCH_SCALAR_ARRAY_HPP_
#define STARTORCH_SCALAR_ARRAY_HPP_

#include <cstdint>

#include "startorch/scalar/reference.hpp"

namespace startorch {

class Array {
 public:

  Reference operator[](uint64_t index);
  ConstReference operator[](uint64_t index) const;

 private:
};

}  // namespace startorch

#endif  // !STARTORCH_SCALAR_ARRAY_HPP_
