#ifndef GUARD_NEURALNET_SUBTRACTX_HPP
#define GUARD_NEURALNET_SUBTRACTX_HPP

#include <poponnx/names.hpp>
#include <poponnx/popx/opx.hpp>

namespace willow {

class SubtractOp;
class SubtractGradOp;

namespace popx {

class SubtractOpx : public Opx {
public:
  SubtractOpx(Op *, Devicex *);
  SubtractOp *getSubtractOp() const;
  virtual void grow() const override final;
};

class SubtractGradOpx : public Opx {
public:
  SubtractGradOpx(Op *, Devicex *);
  SubtractGradOp *getSubtractGradOp() const;
};

} // namespace popx
} // namespace willow

#endif
