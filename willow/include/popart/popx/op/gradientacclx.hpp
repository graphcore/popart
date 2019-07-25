#ifndef GUARD_NEURALNET_GRADIENTACCLX_HPP
#define GUARD_NEURALNET_GRADIENTACCLX_HPP

#include <popart/names.hpp>
#include <popart/popx/opx.hpp>

namespace popart {
namespace popx {

class GradientAcclOpx : public Opx {
public:
  GradientAcclOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
};

class ResetAcclOpx : public Opx {
public:
  ResetAcclOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
};

} // namespace popx
} // namespace popart

#endif
