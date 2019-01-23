#ifndef GUARD_NEURALNET_SUBTRACTX_HPP
#define GUARD_NEURALNET_SUBTRACTX_HPP

#include <poponnx/names.hpp>
#include <poponnx/popx/op/negatex.hpp>
#include <poponnx/popx/op/reducesumx.hpp>
#include <poponnx/popx/opx.hpp>

namespace poponnx {

class SubtractOp;

namespace popx {

class SubtractOpx : public Opx {
public:
  SubtractOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
  InputCreatorType getInputCreatorType(InIndex) const final;
};

// TODO (task T5432) should inherit from ReduceSum when we have numpy
// broadcasting
class SubtractArg0GradOpx : public ReduceSumOpx {
public:
  SubtractArg0GradOpx(Op *, Devicex *);
};

} // namespace popx
} // namespace poponnx

#endif
