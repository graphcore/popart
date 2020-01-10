#ifndef GUARD_NEURALNET_SYNCX_HPP
#define GUARD_NEURALNET_SYNCX_HPP

#include <popart/popx/opx.hpp>

namespace popart {

namespace popx {
class SyncOpx : public Opx {
public:
  SyncOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
};

} // namespace popx
} // namespace popart

#endif
