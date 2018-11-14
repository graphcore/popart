#ifndef GUARD_NEURALNET_SUBTRACTX_HPP
#define GUARD_NEURALNET_SUBTRACTX_HPP

#include <poponnx/names.hpp>
#include <poponnx/popx/identityx.hpp>
#include <poponnx/popx/negatex.hpp>
#include <poponnx/popx/opx.hpp>

namespace willow {

class SubtractOp;

namespace popx {

class SubtractOpx : public Opx {
public:
  SubtractOpx(Op *, Devicex *);
  SubtractOp *getSubtractOp() const;
  virtual void grow() const override final;
};

class SubtractArg0GradOpx : public IdentityOpx {
public:
  SubtractArg0GradOpx(Op *, Devicex *);
};

class SubtractArg1GradOpx : public NegateOpx {
public:
  SubtractArg1GradOpx(Op *, Devicex *);
};

} // namespace popx
} // namespace willow

#endif
