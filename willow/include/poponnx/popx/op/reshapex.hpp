#ifndef GUARD_NEURALNET_RESHAPEX_HPP
#define GUARD_NEURALNET_RESHAPEX_HPP

#include <poponnx/names.hpp>
#include <poponnx/popx/opx.hpp>

namespace poponnx {

class ReshapeOp;
class ReshapeGradOp;

namespace popx {

class ReshapeOpx : public Opx {
public:
  ReshapeOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
  InputCreatorType getInputCreatorType(InIndex) const final;
  poplar::Tensor unwindTensorLayout(poplar::Tensor tensor) const final;
};

// The gradient of a reshape is the reshape in reverse
class ReshapeGradOpx : public ReshapeOpx {
public:
  ReshapeGradOpx(Op *, Devicex *);
};

} // namespace popx
} // namespace poponnx

#endif
