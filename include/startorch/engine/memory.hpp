#ifndef STARTORCH_ENGINE_MEMORY_HPP_
#define STARTORCH_ENGINE_MEMORY_HPP_

namespace startorch {

class Memory {
public:
  explicit operator bool() const;
  bool operator!() const;

private:
};

} // namespace startorch

#endif // !STARTORCH_ENGINE_MEMORY_HPP_
