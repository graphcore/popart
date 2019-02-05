#ifndef GUARD_NEURALNET_DIVX_HPP
#define GUARD_NEURALNET_DIVX_HPP

#include <poponnx/names.hpp>
#include <poponnx/popx/opx.hpp>

namespace poponnx {

class DivOp;

namespace popx {

class DivOpx : public Opx {
public:
  DivOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
  InputCreatorType getInputCreatorType(InIndex) const final;
  poplar::Tensor unwindTensorLayout(poplar::Tensor tensor,
                                    InIndex inIndex,
                                    OutIndex outIndex) const final;
};

} // namespace popx
} // namespace poponnx

#endif
