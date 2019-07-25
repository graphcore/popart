#ifndef GUARD_NEURALNET_SplitX_HPP
#define GUARD_NEURALNET_SplitX_HPP

#include <popart/names.hpp>
#include <popart/popx/opx.hpp>

namespace popart {

namespace popx {

class SplitOpx : public Opx {
public:
  SplitOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
};

} // namespace popx
} // namespace popart

#endif
