#ifndef DARKSIDE_SYSTEM_MACHINE_HPP_
#define DARKSIDE_SYSTEM_MACHINE_HPP_

#include "darkside/memory/arena.hpp"

namespace darkside {

class Machine {};

class HostMachine : public Machine {};

class DeviceMachine : public Machine {};

class MachinePair {
 public:
  MachinePair() = default;
  MachinePair(MachinePair&& other) = default;
  MachinePair(const MachinePair& other) = default;

  MachinePair(Machine* first_machine, Machine* second_machine);

  ~MachinePair() = default;

  MachinePair& operator=(MachinePair&& other) = default;
  MachinePair& operator=(const MachinePair& other) = default;

  void CopyArena(const Arena& source_arena,
                 const Arena& destination_arena) const;
  void CastArena(const Arena& source_arena,
                 const Arena& destination_arena) const;

 private:
  Machine* first_machine_ = nullptr;
  Machine* second_machine_ = nullptr;
};

}  // namespace darkside

#endif  // !DARKSIDE_SYSTEM_MACHINE_HPP_
