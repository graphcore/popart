#ifndef GUARD_NEURALNET_TANHX_HPP
#define GUARD_NEURALNET_TANHX_HPP

#include <poponnx/names.hpp>
#include <poponnx/popx/opx.hpp>

namespace poponnx {

namespace popx {

class TanhOpx : public Opx {
public:
  TanhOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
  InputCreatorType getInputCreatorType(InIndex) const final;
  poplar::Tensor unwindTensorLayout(poplar::Tensor tensor) const final;
};

class TanhGradOpx : public Opx {
public:
  TanhGradOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
};

} // namespace popx
} // namespace poponnx

#endif
