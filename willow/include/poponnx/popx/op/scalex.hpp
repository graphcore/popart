#ifndef GUARD_NEURALNET_SCALEX_HPP
#define GUARD_NEURALNET_SCALEX_HPP

#include <poponnx/names.hpp>
#include <poponnx/popx/op/elementwisex.hpp>

namespace poponnx {

namespace popx {

class ScaleOpx : public ElementWiseUnaryOpx {
public:
  ScaleOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
};

// See T7053 to unify the inplace opxs (TODO)
class ScaleInplaceOpx : public Opx {
public:
  ScaleInplaceOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
  InputCreatorType getInputCreatorType(InIndex) const final;
  poplar::Tensor unwindTensorLayout(poplar::Tensor tensor,
                                    InIndex inIndex,
                                    OutIndex outIndex) const final;
};

class ScaleGradOpx : public ScaleOpx {
public:
  ScaleGradOpx(Op *, Devicex *);
};

} // namespace popx
} // namespace poponnx

#endif
