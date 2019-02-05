#ifndef GUARD_NEURALNET_ELEMENTWISEUNARYX_HPP
#define GUARD_NEURALNET_ELEMENTWISEUNARYX_HPP

#include <poponnx/popx/opx.hpp>

namespace poponnx {
namespace popx {

// Base class for elementwise unary operations
class ElementWiseUnaryOpx : public Opx {
public:
  ElementWiseUnaryOpx(Op *, Devicex *);
  InputCreatorType getInputCreatorType(int index0) const override;
  poplar::Tensor
  unwindTensorLayout(poplar::Tensor tensor, InIndex, OutIndex) const override;
};

} // namespace popx
} // namespace poponnx

#endif
