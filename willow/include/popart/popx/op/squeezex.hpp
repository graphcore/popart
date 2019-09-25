#ifndef GUARD_NEURALNET_SQUEEZEX_HPP
#define GUARD_NEURALNET_SQUEEZEX_HPP

#include <popart/names.hpp>
#include <popart/popx/opx.hpp>

namespace popart {

class SqueezeOp;
class SqueezeGradOp;

namespace popx {

class SqueezeOpx : public Opx {
public:
  SqueezeOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
};

class SqueezeInplaceOpx : public Opx {
public:
  SqueezeInplaceOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
};

class SqueezeGradOpx : public Opx {
public:
  SqueezeGradOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
};

} // namespace popx
} // namespace popart

#endif
