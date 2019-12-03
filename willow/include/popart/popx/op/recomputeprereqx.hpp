#ifndef GUARD_NEURALNET_RECOMPUTEPREREQX_HPP
#define GUARD_NEURALNET_RECOMPUTEPREREQX_HPP

#include <popart/popx/opx.hpp>

namespace popart {
namespace popx {

class RecomputePrereqOpx : public Opx {
public:
  RecomputePrereqOpx(Op *op, Devicex *devicex) : Opx(op, devicex) {}
  void grow(poplar::program::Sequence &) const final {}
};

} // namespace popx
} // namespace popart

#endif
