#ifndef GUARD_NEURALNET_RELUX_HPP
#define GUARD_NEURALNET_RELUX_HPP

#include <poponnx/names.hpp>
#include <poponnx/popx/op/elementwisex.hpp>

namespace poponnx {

class ReluOp;
class ReluInplaceOp;
class ReluGradOp;

namespace popx {

class ReluOpx : public ElementWiseUnaryOpx {
public:
  ReluOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
};

// See T7053 to unify the inplace opxs (TODO)
class ReluInplaceOpx : public Opx {
public:
  ReluInplaceOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
  InputCreatorType getInputCreatorType(InIndex) const final;
  poplar::Tensor unwindTensorLayout(poplar::Tensor tensor,
                                    InIndex inIndex,
                                    OutIndex outIndex) const final;
};

class ReluGradOpx : public Opx {
public:
  ReluGradOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
};

} // namespace popx
} // namespace poponnx

#endif
