#ifndef GUARD_NEURALNET_SQUEEZEX_HPP
#define GUARD_NEURALNET_SQUEEZEX_HPP

#include <poponnx/names.hpp>
#include <poponnx/popx/opx.hpp>

namespace willow {

class SqueezeOp;
class SqueezeGradOp;

namespace popx {

class SqueezeOpx : public Opx {
public:
  SqueezeOpx(Op *, Devicex *);
  SqueezeOp *getSqueezeOp() const;
  void grow(poplar::program::Sequence &) const final;
};

class SqueezeGradOpx : public Opx {
public:
  SqueezeGradOpx(Op *, Devicex *);
  SqueezeGradOp *getSqueezeGradOp() const;
  void grow(poplar::program::Sequence &) const final;
};

} // namespace popx
} // namespace willow

#endif
