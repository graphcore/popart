#ifndef GUARD_NEURALNET_ARGEXTREMAX_HPP
#define GUARD_NEURALNET_ARGEXTREMAX_HPP

#include <poponnx/names.hpp>
#include <poponnx/popx/opx.hpp>

namespace poponnx {

namespace popx {

class ArgExtremaOpx : public Opx {
public:
  ArgExtremaOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;

  poplar::Tensor createInput(InIndex index) const final;
  InputCreatorType getInputCreatorType(InIndex index) const final;
  std::vector<TensorId> mustExistBeforeCreate(InIndex index0) const final;

private:
  unsigned axis;
  bool keepdims;

  virtual poplar::Tensor selectSlice(const poplar::Tensor &sorted,
                                     unsigned axis) const = 0;
};

} // namespace popx
} // namespace poponnx

#endif
