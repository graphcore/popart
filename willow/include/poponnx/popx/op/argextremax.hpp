#ifndef GUARD_NEURALNET_ARGEXTREMAX_HPP
#define GUARD_NEURALNET_ARGEXTREMAX_HPP

#include <poponnx/names.hpp>
#include <poponnx/popx/op/basesortx.hpp>

namespace poponnx {

namespace popx {

class ArgExtremaOpx : public BaseSortOpx {
public:
  ArgExtremaOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;

private:
  bool keepdims;
  virtual poplar::Tensor selectSlice(const poplar::Tensor &sorted,
                                     unsigned axis) const = 0;
};

} // namespace popx
} // namespace poponnx

#endif
