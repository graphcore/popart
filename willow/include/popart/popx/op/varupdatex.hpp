#ifndef GUARD_NEURALNET_VARUPDATEX_HPP
#define GUARD_NEURALNET_VARUPDATEX_HPP

#include <popart/names.hpp>
#include <popart/popx/opx.hpp>

namespace popart {

namespace popx {

class VarUpdateOpx : public Opx {
public:
  VarUpdateOpx(Op *op, Devicex *devicex) : Opx(op, devicex) {}
};

} // namespace popx
} // namespace popart

#endif
