#ifndef GUARD_NEURALNET_ELEMENTWISEUNARYX_HPP
#define GUARD_NEURALNET_ELEMENTWISEUNARYX_HPP

#include <poponnx/popx/opx.hpp>

namespace poponnx {
namespace popx {

// Base class for elementwise unary operations
class ElementWiseUnaryOpx : public Opx {
public:
  ElementWiseUnaryOpx(Op *, Devicex *);
  InputCreatorType getInputCreatorType(InIndex index0) const override;
  poplar::Tensor
  unwindTensorLayout(poplar::Tensor tensor, InIndex, OutIndex) const override;
};

// Base class for elementwise binary operations
class ElementWiseBinaryOpx : public Opx {
public:
  ElementWiseBinaryOpx(Op *, Devicex *);
  InputCreatorType getInputCreatorType(InIndex) const override;
  poplar::Tensor
  unwindTensorLayout(poplar::Tensor tensor, InIndex, OutIndex) const override;
};

} // namespace popx
} // namespace poponnx

#endif
