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
  void grow(poplar::program::Sequence &) const final;
};

// TODO (task T5432) should inherit from ReduceSum when we have numpy
// broadcasting
class SubtractArg0GradOpx : public IdentityOpx {
public:
  SubtractArg0GradOpx(Op *, Devicex *);
};

// TODO (task T5432) should inherit from ReduceSum when we have numpy
// broadcasting
class SubtractArg1GradOpx : public NegateOpx {
public:
  SubtractArg1GradOpx(Op *, Devicex *);
};

} // namespace popx
} // namespace willow

#endif
