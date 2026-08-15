#ifndef DARKSIDE_CORE_VALUE_HPP_
#define DARKSIDE_CORE_VALUE_HPP_

#include "darkside/core/pointer.hpp"
#include "darkside/core/type.hpp"

namespace darkside {

class Pointer;

class Value {
 public:
 private:
  ValueType value_ = false;
  Pointer* pointer_ = nullptr;
};

}  // namespace darkside

#endif  // !DARKSIDE_CORE_VALUE_HPP_
