#ifndef GUARD_NEURALNET_SQUEEZEX_HPP
#define GUARD_NEURALNET_SQUEEZEX_HPP

#include <willow/names.hpp>
#include <willow/popx/opx.hpp>

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
};

} // namespace popx
} // namespace willow

#endif
