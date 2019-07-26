#ifndef GUARD_NEURALNET_ENIGMATIC_HPP
#define GUARD_NEURALNET_ENIGMATIC_HPP

#include <popart/names.hpp>

#include <poplar/OptionFlags.hpp>

// taken directly from enigma, commit
// Date:   Fri Oct 19 13:54:14 2018 +0100

namespace popart {
namespace popx {
namespace enigma {

enum class DeviceType { Cpu, IpuModel, Sim, Hw };

} // namespace enigma
} // namespace popx
} // namespace popart

#endif
