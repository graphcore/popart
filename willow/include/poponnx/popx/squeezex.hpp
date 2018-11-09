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
  virtual void grow() const override final;
};

class SqueezeGradOpx : public Opx {
public:
  SqueezeGradOpx(Op *, Devicex *);
  SqueezeGradOp *getSqueezeGradOp() const;
  virtual void grow() const override final;
};

} // namespace popx
} // namespace willow

#endif
